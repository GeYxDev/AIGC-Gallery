import argparse
import math
from urllib.request import urlopen
from tqdm import tqdm
import sys

sys.path.append('taming-transformers')

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.cuda import get_device_properties

torch.backends.cudnn.benchmark = False  # 加快模型推理速度，可能导致显存溢出
# torch.use_deterministic_algorithms(True)  # 确定性结果，可能导致推理速度变慢

from torch_optimizer import DiffGrad, AdamP
from CLIP import clip
import kornia.augmentation as K
import numpy as np
from PIL import ImageFile, Image, PngImagePlugin

ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings

warnings.filterwarnings('ignore')  # 屏蔽警告输出

# 根据显存和GPU状态决定生成图像的分辨率
default_image_size = 512  # 大于等于8G显存
if not torch.cuda.is_available():  # 无GPU
    default_image_size = 256
elif get_device_properties(0).total_memory <= 2 ** 33:  # 小于8G显存
    default_image_size = 304

# 创建解析器
vq_parser = argparse.ArgumentParser(description='VQGAN-CLIP生成图像')

# 添加参数
vq_parser.add_argument("-p", "--prompts", type=str, help="文本提示", default=None, dest='prompts')
vq_parser.add_argument("-ip", "--image_prompts", type=str, help="图片提示", default=[], dest='image_prompts')
vq_parser.add_argument("-i", "--iterations", type=int, help="迭代次数", default=500, dest='max_iterations')
vq_parser.add_argument("-se", "--save_every", type=int, help="保存轮数", default=50, dest='display_freq')
vq_parser.add_argument("-s", "--size", nargs=2, type=int, help="生成图像分辨率", default=[default_image_size, default_image_size], dest='size')
vq_parser.add_argument("-ii", "--init_image", type=str, help="初始图像", default=None, dest='init_image')
vq_parser.add_argument("-in", "--init_noise", type=str, help="初始噪声图像", default=None, dest='init_noise')
vq_parser.add_argument("-iw", "--init_weight", type=float, help="初始权重", default=0., dest='init_weight')
vq_parser.add_argument("-m", "--clip_model", type=str, help="CLIP模型选择（可选ViT-B/32，ViT-B/16）", default='ViT-B/32', dest='clip_model')
vq_parser.add_argument("-conf", "--vqgan_config", type=str, help="VQGAN设置文件", default=f'checkpoints/vqgan_imagenet_f16_16384.yaml', dest='vqgan_config')
vq_parser.add_argument("-ckpt", "--vqgan_checkpoint", type=str, help="VQGAN检查点文件", default=f'checkpoints/vqgan_imagenet_f16_16384.ckpt', dest='vqgan_checkpoint')
vq_parser.add_argument("-nps", "--noise_prompt_seeds", nargs="*", type=int, help="噪声提示词种子", default=[], dest='noise_prompt_seeds')
vq_parser.add_argument("-npw", "--noise_prompt_weights", nargs="*", type=float, help="噪声提示词权重", default=[], dest='noise_prompt_weights')
vq_parser.add_argument("-lr", "--learning_rate", type=float, help="学习率", default=0.1, dest='step_size')
vq_parser.add_argument("-cutm", "--cut_method", type=str, help="图像裁剪方式", choices=['original', 'updated', 'nrupdated', 'updatedpooling', 'latest'], default='latest', dest='cut_method')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="裁剪后图像数", default=32, dest='cutn')
vq_parser.add_argument("-cutp", "--cut_power", type=float, help="裁剪权重", default=1., dest='cut_pow')
vq_parser.add_argument("-sd", "--seed", type=int, help="种子", default=None, dest='seed')
vq_parser.add_argument("-opt", "--optimiser", type=str, help="优化器", choices=['Adam', 'AdamW', 'Adagrad', 'Adamax', 'DiffGrad', 'AdamP', 'RAdam', 'RMSprop'], default='Adam', dest='optimiser')
vq_parser.add_argument("-o", "--output", type=str, help="输出图像名称", default="output.png", dest='output')
vq_parser.add_argument("-cpe", "--change_prompt_every", type=int, help="提示词改变频率", default=0, dest='prompt_frequency')
vq_parser.add_argument("-d", "--deterministic", action='store_true', help="激活cudnn.deterministic", dest='cudnn_determinism')
vq_parser.add_argument("-aug", "--augments", nargs='+', action='append', type=str, choices=['Ji', 'Sh', 'Gn', 'Pe', 'Ro', 'Af', 'Et', 'Ts', 'Cr', 'Er', 'Re'], help="启用数据增强类型", default=[], dest='augments')
vq_parser.add_argument("-cd", "--cuda_device", type=str, help="使用显卡设备", default="cuda:0", dest='cuda_device')

# 执行parse_args()方法
args = vq_parser.parse_args()

# 预设提示词，文字和图片提示词不存在时使用
if not args.prompts and not args.image_prompts:
    args.prompts = "A painting of an apple in a fruit bowl."

# 激活cudnn.deterministic
if args.cudnn_determinism:
    torch.backends.cudnn.deterministic = True

# 预设数据增强类型，没有指定增强类型时使用
if not args.augments:
    args.augments = [['Af', 'Pe', 'Ji', 'Er']]

# 文字提示词分割成短语
if args.prompts:
    # 分割故事短语
    story_phrases = [phrase.strip() for phrase in args.prompts.split("^")]
    # 建立短语列表
    all_phrases = []
    for phrase in story_phrases:
        all_phrases.append(phrase.split("|"))
    # 设置第一个短语
    args.prompts = all_phrases[0]

# 分割图像提示词
if args.image_prompts:
    args.image_prompts = args.image_prompts.split("|")
    args.image_prompts = [image.strip() for image in args.image_prompts]

# 当CUDA不可用时设置设备为CPU
if not args.cuda_device == 'cpu' and not torch.cuda.is_available():
    args.cuda_device = 'cpu'
    print("警告：未发现CUDA设备，生成速度受到影响。")


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


# 生成初始噪声图像
def random_noise_image(w, h):
    random_image = Image.fromarray(np.random.randint(0, 255, (w, h, 3), dtype=np.dtype('uint8')))
    return random_image


# 生成初始渐变二维图像（灰度）
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


# 生成初始渐变三维图像（彩色）
def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)
    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)
    return result


# 生成具有随机渐变效果的初始图像
def random_gradient_image(w, h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0, 255)),
                        (np.random.randint(1, 255), np.random.randint(2, 255), np.random.randint(3, 128)),
                        (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


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


# 调用自定义替换梯度方法的函数
replace_grad = ReplaceGrad.apply


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


# 调用自定义梯度截断与梯度调整方法的函数
clamp_with_grad = ClampWithGrad.apply


# 向量量化
def vector_quantize(x, codebook):
    # 距离矩阵：输入张量x中每个向量的平方和 + 码本codebook中每个向量的平方和 - 输入张量x和码本codebook的点积
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    # 每个输入向量最近的码本向量的索引
    indices = d.argmin(-1)
    # 量化后向量：转换为独热编码的索引 @ 码本codebook
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
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
        # 调整距离值，防止梯度消失
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


# 解析提示词
def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    # 补充默认值
    vals = vals + ['', '1', '-inf'][len(vals):]
    # 返回提示文本、权重和停止值
    return vals[0], float(vals[1]), float(vals[2])


# 生成多个经过裁剪和增强的图像片段
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.7, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True))
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size, self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1 / .3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size, self.cut_size), scale=(0.1, 1), ratio=(0.75, 1.333), cropping_mode='resample', p=0.5))

        self.augs = nn.Sequential(*augment_list)
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


# 生成多个经过裁剪和增强的图像片段（使用Kornia增强）
class MakeCutoutsPoolingUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=0.7, padding_mode='border'),
            K.RandomPerspective(0.7, p=0.7),
            K.ColorJitter(hue=0.1, saturation=0.1, p=0.7),
            K.RandomErasing((.1, .4), (.3, 1 / .3), same_on_batch=True, p=0.7),
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


# 生成多个经过裁剪和增强的图像片段（可选择Kornia增强，无池化操作）
class MakeCutoutsNRUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

        augment_list = []
        for item in args.augments[0]:
            if item == 'Ji':
                augment_list.append(K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7))
            elif item == 'Sh':
                augment_list.append(K.RandomSharpness(sharpness=0.3, p=0.5))
            elif item == 'Gn':
                augment_list.append(K.RandomGaussianNoise(mean=0.0, std=1., p=0.5))
            elif item == 'Pe':
                augment_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
            elif item == 'Ro':
                augment_list.append(K.RandomRotation(degrees=15, p=0.7))
            elif item == 'Af':
                augment_list.append(K.RandomAffine(degrees=30, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True))
            elif item == 'Et':
                augment_list.append(K.RandomElasticTransform(p=0.7))
            elif item == 'Ts':
                augment_list.append(K.RandomThinPlateSpline(scale=0.8, same_on_batch=True, p=0.7))
            elif item == 'Cr':
                augment_list.append(K.RandomCrop(size=(self.cut_size, self.cut_size), pad_if_needed=True, padding_mode='reflect', p=0.5))
            elif item == 'Er':
                augment_list.append(K.RandomErasing(scale=(.1, .4), ratio=(.3, 1 / .3), same_on_batch=True, p=0.7))
            elif item == 'Re':
                augment_list.append(K.RandomResizedCrop(size=(self.cut_size, self.cut_size), scale=(0.1, 1), ratio=(0.75, 1.333), cropping_mode='resample', p=0.5))

        self.augs = nn.Sequential(*augment_list)
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2: 4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        # 拼接片段成一个批量并应用增强
        batch = self.augs(torch.cat(cutouts, dim=0))
        # 添加噪声
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# 生成多个经过裁剪和增强的图像片段（使用Kornia增强，无池化操作）
class MakeCutoutsUpdate(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augs = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            K.RandomSharpness(0.3, p=0.4),
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2, p=0.4),
        )
        self.noise_fac = 0.1

    def forward(self, input):
        sideY, sideX = input.shape[2: 4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        # 拼接片段成一个批量并应用增强
        batch = self.augs(torch.cat(cutouts, dim=0))
        # 添加噪声
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


# 生成多个经过裁剪的图像片段（无池化操作）
class MakeCutoutsOrig(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2: 4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety: offsety + size, offsetx: offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
        # 返回拼接后的片段，前向传播中梯度截断至[0, 1]
        return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)


# 加载预训练的VQGAN模型
def load_vqgan_model(config_path, checkpoint_path):
    # 标记是否加载了GumbelVQ模型
    global gumbel
    gumbel = False
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        # 模型为评估模式且禁用梯度
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.vqgan.GumbelVQ':
        model = vqgan.GumbelVQ(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
        gumbel = True
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'不明模型类型: {config.model.target}')
    # 删除模型损失函数，无需计算模型损失
    del model.loss
    return model


# 调整图像大小
def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
    return image.resize(size, Image.LANCZOS)


# 使用VQGAN-CLIP生成图像
device = torch.device(args.cuda_device)
# 加载VQGAN
model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
# 根据Pytorch版本决定是否使用JIT编译
jit = True if "1.7.1" in torch.__version__ else False
# 加载CLIP，评估模式且禁用梯度
perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(device)

# CLIP视觉部分的输入分辨率，决定VQGAN生成图像的裁片大小
cut_size = perceptor.visual.input_resolution
# 缩放因子，缩放图像至VQGAN解码器的输入分辨率
f = 2 ** (model.decoder.num_resolutions - 1)

# 选择图像裁剪和增强方法
if args.cut_method == 'latest':
    make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)
elif args.cut_method == 'original':
    make_cutouts = MakeCutoutsOrig(cut_size, args.cutn, cut_pow=args.cut_pow)
elif args.cut_method == 'updated':
    make_cutouts = MakeCutoutsUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)
elif args.cut_method == 'nrupdated':
    make_cutouts = MakeCutoutsNRUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)
else:
    make_cutouts = MakeCutoutsPoolingUpdate(cut_size, args.cutn, cut_pow=args.cut_pow)

# 根据目标图像宽度和高度计算token数量
toksX, toksY = args.size[0] // f, args.size[1] // f
# 实际生成图像的宽度和高度（生成图像尺寸是f的整数倍）
sideX, sideY = toksX * f, toksY * f

# 是否使用Gumbel软化机制
if gumbel:
    e_dim = 256
    n_toks = model.quantize.n_embed
    z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

# 若使用初始图像
if args.init_image:
    if 'http' in args.init_image:
        img = Image.open(urlopen(args.init_image))
    else:
        img = Image.open(args.init_image)
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    # 使用VQGAN编码器处理，归一化：[0, 1]转换至[-1, 1]
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
# 若使用初始噪声图像
elif args.init_noise == 'pixels':
    img = random_noise_image(args.size[0], args.size[1])
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
# 若使用初始渐变图像
elif args.init_noise == 'gradient':
    img = random_gradient_image(args.size[0], args.size[1])
    pil_image = img.convert('RGB')
    pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
# 生成初始隐向量Z
else:
    # 随机索引
    one_hot = F.one_hot(torch.randint(n_toks, [toksY * toksX], device=device), n_toks).float()
    # 根据Gumbel软化机制的使用情况，使用模型的嵌入权重生成隐向量Z
    if gumbel:
        z = one_hot @ model.quantize.embed.weight
    else:
        z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)

z_orig = z.clone()
# 隐向量Z可训练
z.requires_grad_(True)

# 归一化变换
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

# 列表存储编码后提示词
pMs = []
# 文本提示
if args.prompts:
    for prompt in args.prompts:
        # 分割提示文本
        txt, weight, stop = split_prompt(prompt)
        # 编码文本为文本嵌入向量
        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
        pMs.append(Prompt(embed, weight, stop).to(device))
# 图像提示
for prompt in args.image_prompts:
    path, weight, stop = split_prompt(prompt)
    img = Image.open(path)
    pil_image = img.convert('RGB')
    img = resize_image(pil_image, (sideX, sideY))
    # 生成裁剪片段
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
    # 编码图像为图像嵌入向量
    embed = perceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))
# 噪声提示
for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    # 生成标准正态分布的噪声并填充
    pMs.append(Prompt(embed, weight).to(device))


# 设置优化器优化隐向量Z
def get_opt(opt_name, opt_lr):
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9)
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)
    elif opt_name == "RAdam":
        opt = optim.RAdam([z], lr=opt_lr)
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    else:
        print("警告：未知优化器选择，设置为默认优化器Adam。")
        opt = optim.Adam([z], lr=opt_lr)
    return opt


# 获得优化器
opt = get_opt(args.optimiser, args.step_size)

print('使用设备：', device)
print('优化器：', args.optimiser)
if args.prompts:
    print('文本提示词：', args.prompts)
if args.image_prompts:
    print('图像提示词：', args.image_prompts)
if args.init_image:
    print('初始图像：', args.init_image)
if args.noise_prompt_weights:
    print('噪声提示权重：', args.noise_prompt_weights)

# 是否提供随机种子
if args.seed is None:
    seed = torch.seed()
else:
    seed = args.seed
torch.manual_seed(seed)
print('随机种子：', seed)


# 隐向量Z矢量量化并生成图像
def synth(z):
    # 根据Gumbel软化机制的使用情况，选择矢量量化使用的权重
    if gumbel:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(3, 1)
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(3, 1)
    # 解码生成图像，在反向传播中返回原始梯度
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


# 记录训练状态（推理模式）
@torch.inference_mode()
def checkin(i, losses):
    losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
    tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
    # 生成图像
    out = synth(z)
    info = PngImagePlugin.PngInfo()
    info.add_text('comment', f'{args.prompts}')
    # 保存图像到指定路径
    TF.to_pil_image(out[0].cpu()).save(args.output, pnginfo=info)


# 计算图像与文本提示的匹配程度
def ascend_txt():
    global i
    # 生成图像
    out = synth(z)
    # 生成多个裁剪片段并编码为图像嵌入向量
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()
    # 存放损失
    result = []
    # 若定义了初始权重
    if args.init_weight:
        # 计算隐向量Z与零张量的均方误差损失，根据迭代次数逐渐减小损失权重：正则化项，稳定训练
        result.append(F.mse_loss(z, torch.zeros_like(z_orig)) * ((1 / torch.tensor(i * 2 + 1)) * args.init_weight) / 2)
    # 计算有关提示的嵌入向量与生成图像裁剪片段的图像嵌入向量之间的损失，损失权重不变
    for prompt in pMs:
        result.append(prompt(iii))
    # 返回损失
    return result


# 训练生图
def train(i):
    # 梯度清零
    opt.zero_grad(set_to_none=True)
    # 计算损失
    lossAll = ascend_txt()
    # 显示结果
    if i % args.display_freq == 0:
        checkin(i, lossAll)
    # 损失求和
    loss = sum(lossAll)
    # 反向传播
    loss.backward()
    # 优化隐向量Z
    opt.step()

    # 在没有梯度的情况下限制隐向量Z至合理范围
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))


i = 0  # 迭代次数计数
p = 1  # 文字提示计数

# 图像生成过程
try:
    with tqdm() as pbar:
        while True:
            # 轮换更改文本提示
            if args.prompt_frequency > 0:
                if i % args.prompt_frequency == 0 and i > 0:
                    # 没有足够的文本提示时循环使用
                    if p >= len(all_phrases):
                        p = 0
                    # 再次初始化存储编码后提示词的列表
                    pMs = []
                    # 使用索引p对应的文本提示
                    args.prompts = all_phrases[p]
                    print('文本提示词：', args.prompts)
                    # 编码文本为文本嵌入向量
                    for prompt in args.prompts:
                        txt, weight, stop = split_prompt(prompt)
                        embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
                        pMs.append(Prompt(embed, weight, stop).to(device))
                    # 指向下一条文本提示
                    p += 1

            # 训练生图
            train(i)

            # 达到最大迭代次数后停止训练
            if i == args.max_iterations:
                break
            # 完成一次迭代
            i += 1
            pbar.update()
# 中断时不输出错误信息
except KeyboardInterrupt:
    pass
