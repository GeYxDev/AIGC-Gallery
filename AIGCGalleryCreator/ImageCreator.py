import math
import os.path
import sys
from datetime import datetime

import imageio
import numpy as np
from tqdm import tqdm

sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from pytorch_pretrained_biggan import BigGAN, convert_to_images
from scipy.stats import truncnorm
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties

torch.backends.cudnn.benchmark = False  # 加快模型推理速度，可能导致显存溢出

from CLIP import clip
import kornia.augmentation as K
from PIL import ImageFile, PngImagePlugin

ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings

warnings.filterwarnings('ignore')  # 屏蔽警告输出

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 根据显存和GPU状态决定生成图像的分辨率
image_size = 512  # 大于等于8G显存
if not torch.cuda.is_available() or get_device_properties(0).total_memory <= 2 ** 33:  # 无GPU或小于8G显存
    image_size = 256


# sinc函数
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


# Lanczos滤波器
def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


# 斜坡函数
def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1: -1]


# 重采样
def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size
    input = input.view([n * c, 1, h, w])
    # 高度方向降采样
    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])
    # 宽度方向降采样
    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])
    input = input.view([n, c, h, w])
    return F.interpolate(input, size, mode='bicubic', align_corners=align_corners)


# 自定义替换梯度方法
class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        # 保存反向传播的输入值的形状至上下文对象中
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        # 输入的梯度调整到反向传播的输入值的形状
        return None, grad_in.sum_to_size(ctx.shape)


# 自定义梯度截断与梯度调整方法
class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        # 输入张量保存到上下文对象中
        ctx.save_for_backward(input)
        # 返回截断后的输入张量
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        # 计算输入张量与截断后的输入张量的差值，差值与输入梯度相乘，若乘积大于零则保留，否则设为0
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None


# 向量量化
def vector_quantize(x, codebook):
    # 距离矩阵：输入张量x中每个向量的平方和 + 码本codebook中每个向量的平方和 - 输入张量x和码本codebook的点积
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    # 每个输入向量最近的码本向量的索引
    indices = d.argmin(-1)
    # 量化后向量：转换为独热编码的索引 @ 码本codebook
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    # 调用自定义替换梯度方法的函数
    replace_grad = ReplaceGrad.apply
    # 替换梯度，在反向传播中返回原始输入x的梯度
    return replace_grad(x_q, x)


# 嵌入向量与输入张量的相似度计算
class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        # 计算角距离
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        # 距离值乘以权重的符号
        dists = dists * self.weight.sign()
        # 调用自定义替换梯度方法的函数
        replace_grad = ReplaceGrad.apply
        # 调整距离值，防止梯度消失
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


# 解析提示词
def split_prompt(prompt):
    vals = prompt.rsplit(':', 1)
    # 补充默认值
    vals = vals + ['', '1'][len(vals):]
    # 返回提示文本和权重
    return vals[0], float(vals[1])


# 生成多个经过裁剪和增强的图像片段
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True),
            K.RandomPerspective(distortion_scale=0.7, p=0.7),
            K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7),
            K.RandomErasing(scale=(.1, .4), ratio=(.3, 1 / .3), same_on_batch=True, p=0.7)
        )
        self.noise_fac = 0.1
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input):
        cutouts = []
        for _ in range(self.cutn):
            cutout = (self.av_pool(input) + self.max_pool(input)) / 2
            cutouts.append(cutout)
        # 拼接片段成一个批量并应用增强
        batch = self.augs(torch.cat(cutouts, dim=0))
        # 添加噪声
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# 加载预训练的VQGAN模型
def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    model = vqgan.VQModel(**config.model.params)
    # 模型为评估模式且禁用梯度
    model.eval().requires_grad_(False)
    model.init_from_ckpt(checkpoint_path)
    # 删除模型损失函数，无需计算模型损失
    del model.loss
    return model


# 加载ChineseCLIP模型
def load_clip_model(clip_model):
    # 根据Pytorch版本决定是否使用JIT编译
    jit = True if "1.7.1" in torch.__version__ else False
    # 加载CLIP，评估模式且禁用梯度
    model = clip.load(clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)
    return model


# VQGAN-CLIP图像生成
class VQGANImageCreator:
    def __init__(self, args):
        self.args = args
        # 当前迭代次数
        self.cur_iter = 0
        # 加载预训练的VQGAN模型
        self.model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
        # 加载预训练的CLIP模型
        self.perceptor = load_clip_model(args.clip_model).to(device)
        # CLIP视觉部分的输入分辨率，决定VQGAN生成图像的裁片大小
        cut_size = self.perceptor.visual.input_resolution
        # 应用图像裁剪和增强方法
        self.make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
        # 缩放因子，缩放图像至VQGAN解码器的输入分辨率
        f = 2 ** (self.model.decoder.num_resolutions - 1)
        # 根据目标图像宽度和高度计算token数量
        toksX, toksY = image_size // f, image_size // f
        # 获得码本中向量权重范围
        e_dim = self.model.quantize.e_dim
        n_toks = self.model.quantize.n_e
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        # 使用模型的嵌入权重生成隐向量Z
        one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
        self.z = one_hot @ self.model.quantize.embedding.weight
        self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        # 隐向量Z可训练
        self.z.requires_grad_(True)
        # 设置优化器
        self.opt = optim.Adam([self.z], lr=args.learning_rate)
        # 归一化变换
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        # 列表存储编码后提示词
        self.pMs = []
        # 编码文本为文本嵌入向量
        for prompt in args.prompts:
            # 分割提示文本
            txt, weight = split_prompt(prompt)
            # 编码文本为文本嵌入向量
            embed = self.perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight).to(device))
        # 噪声提示
        for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(generator=gen)
            # 生成标准正态分布的噪声并填充
            self.pMs.append(Prompt(embed, weight).to(device))

    # 隐向量Z矢量量化并生成图像
    def synth(self, z):
        # 调用自定义梯度截断与梯度调整方法的函数
        clamp_with_grad = ClampWithGrad.apply
        # 矢量量化使用的权重
        z_q = vector_quantize(z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        # 解码生成图像，在反向传播中返回原始梯度
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    # 记录训练状态（推理模式）
    @torch.inference_mode()
    def checkin(self, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {self.cur_iter}, loss: {sum(losses).item():g}, losses: {losses_str}')
        # 生成图像
        out = self.synth(self.z)
        info = PngImagePlugin.PngInfo()
        info.add_text('comment', f'{self.args.prompts}')
        # 保存图像到指定路径
        TF.to_pil_image(out[0].cpu()).save(os.path.join(self.args.output, f'{datetime.now().timestamp() * 1000}.png'), pnginfo=info)

    # 计算图像与文本提示的匹配程度
    def ascend_txt(self):
        # 生成图像
        out = self.synth(self.z)
        # 生成多个裁剪片段并编码为图像嵌入向量
        image_embed = self.perceptor.encode_image(self.normalize(self.make_cutouts(out))).float()
        # 存放损失
        result = []
        # 计算有关提示的嵌入向量与生成图像裁剪片段的图像嵌入向量之间的损失，损失权重不变
        for prompt in self.pMs:
            result.append(prompt(image_embed))
        # 返回损失
        return result

    # 单次生图
    def train_once(self):
        # 梯度清零
        self.opt.zero_grad(set_to_none=True)
        # 计算损失
        lossAll = self.ascend_txt()
        # 存储中间结果
        if self.cur_iter % self.args.save_every == 0:
            self.checkin(lossAll)
        # 损失求和
        loss = sum(lossAll)
        # 反向传播
        loss.backward()
        # 优化隐向量Z
        self.opt.step()
        # 在没有梯度的情况下限制隐向量Z至合理范围
        with torch.inference_mode():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    # 训练生图
    def train(self, generateStatus):
        # 当前迭代次数
        self.cur_iter = 0
        try:
            with tqdm(self.args.max_iterations) as pbar:
                isInterrupt = generateStatus['interrupt']
                while not isInterrupt:
                    self.train_once()
                    self.cur_iter += 1
                    if self.cur_iter == self.args.max_iterations:
                        break
                    pbar.update()
                    isInterrupt = generateStatus['interrupt']
                generateStatus['currentIter'] = self.cur_iter
        except KeyboardInterrupt:
            pass

    # 获得当前迭代次数
    def getCurIter(self):
        return self.cur_iter


# BigGAN-CLIP图像生成
class BigGANImageCreator:
    def __init__(self, args):
        self.args = args
        # 当前迭代次数
        self.cur_iter = 0
        # 加载预训练的BigGAN模型
        self.model = BigGAN.from_pretrained(f'biggan-deep-{image_size}').cuda().eval()
        # 加载预训练的CLIP模型
        self.perceptor = load_clip_model(args.clip_model).to(device)
        # 设置图像大小
        self.sideX, self.sideY = image_size, image_size
        # 生成随机噪声向量
        seed = None
        state = None if seed is None else np.random.RandomState(seed)
        np.random.seed(seed)
        self.noise_vector = truncnorm.rvs(-2 * args.truncation, 2 * args.truncation, size=(1, 128), random_state=state).astype(np.float32)
        # 初始化类向量
        self.class_vector = np.random.rand(1, 1000).astype(np.float32)
        self.eps = 1e-8
        self.class_vector = np.log(self.class_vector + self.eps)
        self.noise_vector = torch.tensor(self.noise_vector, requires_grad=True, device=device)
        self.class_vector = torch.tensor(self.class_vector, requires_grad=True, device=device)
        # 反向传播时对类向量和噪声向量进行更新
        params = [self.noise_vector]
        params += [self.class_vector]
        # 设置优化器
        self.optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        # 编码文本为文本嵌入向量
        self.embed = self.perceptor.encode_text(clip.tokenize(args.prompts).to(device)).float()
        # CLIP视觉部分的输入分辨率，决定BigGAN生成图像的裁片大小
        self.cut_size = self.perceptor.visual.input_resolution
        # 归一化变换
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # 计算图像与文本提示的匹配程度
    def ascend_txt(self):
        # 限制噪声向量范围
        noise_vector_trunc = self.noise_vector.clamp(-2 * self.args.truncation, 2 * self.args.truncation)
        # 对类向量进行归一化操作并将其转换为概率分布
        class_vector_norm = torch.nn.functional.softmax(self.class_vector)
        # 生成图像
        out = self.model(noise_vector_trunc, class_vector_norm, self.args.truncation)
        # 随机裁剪图像
        max_size = min(self.sideX, self.sideY)
        min_size = min(self.sideX, self.sideY, self.cut_size)
        cutouts = []
        for _ in range(self.args.cutn):
            size = int(torch.rand([]) * (max_size - min_size) + min_size)
            # 设定随机裁剪的起点
            offset_x = torch.randint(0, self.sideX - size + 1, ())
            offset_y = torch.randint(0, self.sideY - size + 1, ())
            cutout = out[:, :, offset_y: offset_y + size, offset_x: offset_x + size]
            # 缩放随机裁剪后的图像至CLIP输入大小
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        batch = torch.cat(cutouts, dim=0)
        # 编码图片为图片嵌入向量
        image_embed = self.perceptor.encode_image(self.normalize(batch))
        # 使用余弦距离作为损失函数
        factor = 100
        loss = factor * (1 - torch.cosine_similarity(image_embed, self.embed, dim=-1).mean())
        total_loss = loss
        # 迭代过程中约束类向量
        reg = -factor * self.args.class_ent_reg * (class_vector_norm * torch.log(class_vector_norm + self.eps)).sum()
        total_loss += reg
        if self.cur_iter % self.args.save_every == 0:
            # 打印损失
            tqdm.write(f'i: {self.cur_iter}, loss: {total_loss.item():g}, losses: {total_loss.item():g}')
            # 保存图像到指定路径
            with torch.no_grad():
                image = out.cpu().numpy()
            image = convert_to_images(image)[0]
            imageio.imwrite(os.path.join(self.args.output, f'{datetime.now().timestamp() * 1000}.png'), np.asarray(image))
        return total_loss

    # 训练生图
    def train(self, generateStatus):
        # 当前迭代次数
        self.cur_iter = 0
        try:
            with tqdm(self.args.max_iterations) as pbar:
                isInterrupt = generateStatus['interrupt']
                while not isInterrupt:
                    self.optimizer.zero_grad()
                    loss = self.ascend_txt()
                    loss.backward()
                    self.optimizer.step()
                    self.cur_iter += 1
                    if self.cur_iter == self.args.max_iterations:
                        break
                    pbar.update()
                    isInterrupt = generateStatus['interrupt']
                generateStatus['currentIter'] = self.cur_iter
        except KeyboardInterrupt:
            pass

    # 获得当前迭代次数
    def getCurIter(self):
        return self.cur_iter
