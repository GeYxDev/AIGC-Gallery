import argparse
import math

import imageio
import numpy as np
import torch
from torch.nn import functional as F
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, convert_to_images
from scipy.stats import truncnorm, dirichlet
from torch.cuda import get_device_properties
from torchvision.transforms import transforms
from tqdm import tqdm

from CLIP import clip

import warnings

warnings.filterwarnings('ignore')  # 屏蔽警告输出

# 根据显存和GPU状态决定生成图像的分辨率
image_size = 512  # 大于等于8G显存
if not torch.cuda.is_available() or get_device_properties(0).total_memory <= 2 ** 33:  # 无GPU或小于8G显存
    image_size = 256

# 创建解析器
vq_parser = argparse.ArgumentParser(description='BigGAN-CLIP生成图像')

# 添加参数
vq_parser.add_argument("-p", "--prompts", type=str, help="文本提示", default=None, dest='prompts')
vq_parser.add_argument("-i", "--iterations", type=int, help="迭代次数", default=500, dest='max_iterations')
vq_parser.add_argument("-se", "--save_every", type=int, help="保存轮数", default=50, dest='display_freq')
vq_parser.add_argument("-m", "--clip_model", type=str, help="CLIP模型选择（可选ViT-B/32，ViT-B/16）", default='ViT-B/32', dest='clip_model')
vq_parser.add_argument("-lr", "--learning_rate", type=float, help="学习率", default=0.1, dest='step_size')
vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="裁剪后图像数", default=32, dest='cutn')
vq_parser.add_argument("-ic", "--initial_class", type=str, help="类向量初始化", default='Random mix', dest='initial_class')
vq_parser.add_argument("-oc", "--optimize_class", type=bool, help="类向量优化", default=True, dest='optimize_class')
vq_parser.add_argument("-cs", "--class_smoothing", type=float, help="类向量平滑", default=0.1, dest='class_smoothing')
vq_parser.add_argument("-t", "--truncation", type=int, help="噪声截断阈值", default=1, dest='truncation')
vq_parser.add_argument("-ce", "--class_ent_reg", type=float, help="类向量正则化", default=0.0001, dest='class_ent_reg')
vq_parser.add_argument("-o", "--output", type=str, help="输出图像名称", default="output.png", dest='output')
vq_parser.add_argument("-cd", "--cuda_device", type=str, help="使用显卡设备", default="cuda:0", dest='cuda_device')

# 执行parse_args()方法
args = vq_parser.parse_args()

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

# 加载预训练的BigGAN模型
model = BigGAN.from_pretrained(f'biggan-deep-{image_size}').cuda().eval()
# 根据Pytorch版本决定是否使用JIT编译
jit = True if "1.7.1" in torch.__version__ else False
# 加载CLIP，评估模式且禁用梯度
perceptor = clip.load(args.clip_model, jit=jit)[0].eval().requires_grad_(False).to(args.cuda_device)

# 设置图像大小
sideX, sideY = image_size, image_size

# 生成随机向量
seed = None
state = None if seed is None else np.random.RandomState(seed)
np.random.seed(seed)
noise_vector = truncnorm.rvs(-2 * args.truncation, 2 * args.truncation, size=(1, 128), random_state=state).astype(np.float32)

# 生成类向量
if args.initial_class.lower() == 'random class':
    class_vector = np.ones(shape=(1, 1000), dtype=np.float32) * args.class_smoothing / 999
    class_vector[0, np.random.randint(1000)] = 1 - args.class_smoothing
elif args.initial_class.lower() == 'random dirichlet':
    class_vector = dirichlet.rvs([1 / 1000] * 1000, size=1, random_state=state).astype(np.float32)
elif args.initial_class.lower() == 'random mix':
    class_vector = np.random.rand(1, 1000).astype(np.float32)
else:
    if args.initial_class.lower() == 'from prompt':
        initial_class = args.prompts
    class_vector = one_hot_from_names(args.initial_class, batch_size=1)
    class_vector = class_vector * (1 - args.class_smoothing * 1000 / 999) + args.class_smoothing / 999
eps = 1e-8
class_vector = np.log(class_vector + eps)

# 将噪声向量和类向量转化为张量
noise_vector = torch.tensor(noise_vector, requires_grad=True, device=args.cuda_device)
class_vector = torch.tensor(class_vector, requires_grad=True, device=args.cuda_device)

# 类向量优化
params = [noise_vector]
if args.optimize_class:
    params += [class_vector]
opt = torch.optim.Adam(params, lr=args.step_size)

# 编码文本为文本嵌入向量
embed = perceptor.encode_text(clip.tokenize(args.prompts).to(args.cuda_device)).float()

# CLIP视觉部分的输入分辨率，决定BigGAN生成图像的裁片大小
cut_size = perceptor.visual.input_resolution

# 归一化变换
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

# 计算图像与文本提示的匹配程度
def ascend_txt(i):
    # 限制噪声向量范围
    noise_vector_trunc = noise_vector.clamp(-2 * args.truncation, 2 * args.truncation)
    # 对类向量进行归一化操作并将其转换为概率分布
    class_vector_norm = torch.nn.functional.softmax(class_vector)
    # 生成图像
    out = model(noise_vector_trunc, class_vector_norm, args.truncation)
    # 随机裁剪图像
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, cut_size)
    cutouts = []
    for _ in range(args.cutn):
        size = int(torch.rand([]) * (max_size - min_size) + min_size)
        # 设定随机裁剪的起点
        offset_x = torch.randint(0, sideX - size + 1, ())
        offset_y = torch.randint(0, sideY - size + 1, ())
        cutout = out[:, :, offset_y: offset_y + size, offset_x: offset_x + size]
        # 缩放随机裁剪后的图像至CLIP输入大小
        cutouts.append(resample(cutout, (cut_size, cut_size)))
    batch = torch.cat(cutouts, dim=0)
    # 编码图片为图片嵌入向量
    image_embed = perceptor.encode_image(normalize(batch))
    # 使用余弦距离作为损失函数
    factor = 100
    loss = factor * (1 - torch.cosine_similarity(image_embed, embed, dim=-1).mean())
    total_loss = loss
    # 迭代过程中约束类向量
    if args.optimize_class and args.class_ent_reg:
        reg = -factor * args.class_ent_reg * (class_vector_norm * torch.log(class_vector_norm + eps)).sum()
        total_loss += reg
    if i % args.display_freq == 0:
        # 打印损失
        tqdm.write(f'i: {i}, loss: {total_loss.item():g}')
        # 保存图像到指定路径
        with torch.no_grad():
            image = out.cpu().numpy()
        image = convert_to_images(image)[0]
        imageio.imwrite(args.output, np.asarray(image))
    return total_loss

# 训练生图
def train(i):
    # 梯度清零
    opt.zero_grad(set_to_none=True)
    # 计算损失
    loss = ascend_txt(i)
    # 反向传播
    loss.backward()
    # 优化向量
    opt.step()

i = 0  # 迭代次数计数

# 图像生成过程
try:
    with tqdm() as pbar:
        while True:
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
