import functools
import math

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F

import layers
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d


# 优化器
class Adam16(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        super(Adam16, self).__init__(params, defaults)

    def load_state_dict(self, state_dict):
        super(Adam16, self).load_state_dict(state_dict)
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
                self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
                self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
                    state['fp32_p'] = p.data.float()
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], state['fp32_p'])
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
                p.data = state['fp32_p'].half()
        return loss


# 生成器卷积块
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv,
                 which_bn=layers.bn, activation=None, up_sample=None, channel_ratio=4):
        super(GBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.in_channels // channel_ratio
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels,kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels,kernel_size=1, padding=0)
        self.bn1 = self.which_bn(self.in_channels)
        self.bn2 = self.which_bn(self.hidden_channels)
        self.bn3 = self.which_bn(self.hidden_channels)
        self.bn4 = self.which_bn(self.hidden_channels)
        self.up_sample = up_sample

    def forward(self, x, y):
        h = self.conv1(self.activation(self.bn1(x, y)))
        h = self.activation(self.bn2(h, y))
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
        if self.up_sample:
            h = self.up_sample(h)
            x = self.up_sample(x)
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x


# 生成器输出分辨率与网络配置
def G_arch(ch=64, attention='64'):
    arch = {
        256: {
            'in_channels': [ch * item for item in [16, 16, 8, 8, 4, 2]],
            'out_channels': [ch * item for item in [16, 8, 8, 4, 2, 1]],
            'up_sample': [True] * 6,
            'resolution': [8, 16, 32, 64, 128, 256],
            'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 9)}
        },
        128: {
            'in_channels': [ch * item for item in [16, 16, 8, 4, 2]],
            'out_channels': [ch * item for item in [16, 8, 4, 2, 1]],
            'up_sample': [True] * 5,
            'resolution': [8, 16, 32, 64, 128],
            'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 8)}
        },
        64: {
            'in_channels': [ch * item for item in [16, 16, 8, 4]],
            'out_channels': [ch * item for item in [16, 8, 4, 2]],
            'up_sample': [True] * 4,
            'resolution': [8, 16, 32, 64],
            'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 7)}
        },
        32: {
            'in_channels': [ch * item for item in [4, 4, 4]],
            'out_channels': [ch * item for item in [4, 4, 4]],
            'up_sample': [True] * 3,
            'resolution': [8, 16, 32],
            'attention': {2 ** i: (2 ** i in [int(item) for item in attention.split('_')]) for i in range(3, 6)}
        }
    }
    return arch


# 生成器
class Generator(nn.Module):
    def __init__(self, G_ch=64, G_depth=2, dim_z=128, bottom_width=4, resolution=128, G_kernel_size=3, G_attn='64',
                 n_classes=1000, num_G_SVs=1, num_G_SV_itrs=1, G_shared=True, shared_dim=0, hier=False,
                 cross_replica=False, mybn=False, G_activation=nn.ReLU(inplace=False), G_lr=5e-5, G_B1=0.0, G_B2=0.999,
                 adam_eps=1e-8, BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False, G_init='ortho',
                 skip_init=False, no_optim=False, G_param='SN', norm_style='bn'):
        super(Generator, self).__init__()
        # 通道宽度单位
        self.ch = G_ch
        # 每阶段残差块数量
        self.G_depth = G_depth
        # 潜在空间维度
        self.dim_z = dim_z
        # 初始空间维度
        self.bottom_width = bottom_width
        # 输出分辨率
        self.resolution = resolution
        # 卷积核大小
        self.kernel_size = G_kernel_size
        # 注意力机制
        self.attention = G_attn
        # 类别数
        self.n_classes = n_classes
        # 共享嵌入
        self.G_shared = G_shared
        # 共享嵌入维度
        self.shared_dim = shared_dim if shared_dim > 0 else dim_z
        # 层次化潜在空间
        self.hier = hier
        # 跨副本批归一化
        self.cross_replica = cross_replica
        # 使用特制批归一化
        self.mybn = mybn
        # 残差块非线性激活函数
        self.activation = G_activation
        # 初始化风格
        self.init = G_init
        # 参数化风格
        self.G_param = G_param
        # 归一化风格
        self.norm_style = norm_style
        # 批归一化epsilon
        self.BN_eps = BN_eps
        # 谱归一化epsilon
        self.SN_eps = SN_eps
        # 使用fp16
        self.fp16 = G_fp16
        # 模型结构字典
        self.arch = G_arch(self.ch, self.attention)[resolution]
        # 参数计数
        self.param_count = 0
        # 卷积和归一化类型选择
        if self.G_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d, kernel_size=3, padding=1,
                                                num_svs=num_G_SVs, num_itrs=num_G_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=num_G_SVs,
                                                  num_itrs=num_G_SV_itrs, eps=self.SN_eps)
        else:
            self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
            self.which_linear = nn.Linear
        # 定义嵌入和批归一化操作
        self.which_embedding = nn.Embedding
        bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared else self.which_embedding)
        self.which_bn = functools.partial(layers.ccbn, which_linear=bn_linear, cross_replica=self.cross_replica,
                                          mybn=self.mybn, input_size=(self.shared_dim + self.dim_z
                                                                      if self.G_shared else self.n_classes),
                                          norm_style=self.norm_style, eps=self.BN_eps)
        # 预训练模型
        self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared else layers.identity())
        # 第一层线性层
        self.linear = self.which_linear(self.dim_z + self.shared_dim,
                                        self.arch['in_channels'][0] * (self.bottom_width ** 2))
        # 构建后续模型结构
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                GBlock(in_channels=self.arch['in_channels'][index],
                    out_channels=self.arch['in_channels'][index] if g_index == 0
                    else self.arch['out_channels'][index],
                    which_conv=self.which_conv,
                    which_bn=self.which_bn,
                    activation=self.activation,
                    up_sample=(functools.partial(F.interpolate, scale_factor=2)
                               if self.arch['up_sample'][index] and g_index == (self.G_depth - 1) else None))
            ] for g_index in range(self.G_depth)]
            # 使用注意力模块
            if self.arch['attention'][self.arch['resolution'][index]]:
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # 输出层
        self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                    cross_replica=self.cross_replica, mybn=self.mybn),
                                          self.activation, self.which_conv(self.arch['out_channels'][-1], 3))
        # 初始化模型权重
        if not skip_init:
            self.init_weights()
        # 设置优化器
        if no_optim:
            return
        self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
        if G_mixed_precision:
            self.optim = Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0,
                                eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0,
                                    eps=self.adam_eps)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized.')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])

    def forward(self, z, y):
        if self.hier:
            z = torch.cat([y, z], 1)
            y = z
        h = self.linear(z)
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        for index, blocklist in enumerate(self.blocks):
            for block in [blocklist]:
                h = block(h, y)
        return torch.tanh(self.output_layer(h))


# 判别器卷积块
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=layers.SNConv2d, pre_activation=True,
                 activation=None, down_sample=None, channel_ratio=4):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hidden_channels = self.out_channels // channel_ratio
        self.which_conv = which_conv
        self.pre_activation = pre_activation
        self.activation = activation
        self.down_sample = down_sample
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=1, padding=0)
        self.conv2 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv3 = self.which_conv(self.hidden_channels, self.hidden_channels)
        self.conv4 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=1, padding=0)
        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels - in_channels, kernel_size=1, padding=0)

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = torch.cat([x, self.conv_sc(x)], 1)
        return x

    def forward(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(self.activation(h))
        h = self.conv3(self.activation(h))
        h = self.activation(h)
        if self.downsample:
            h = self.downsample(h)
        h = self.conv4(h)
        return h + self.shortcut(x)


# 判别器输出分辨率与网络配置
def D_arch(ch=64, attention='64'):
    arch = {
        256: {
            'in_channels': [item * ch for item in [1, 2, 4, 8, 8, 16]],
            'out_channels': [item * ch for item in [2, 4, 8, 8, 16, 16]],
            'downsample': [True] * 6 + [False],
            'resolution': [128, 64, 32, 16, 8, 4, 4],
            'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')] for i in range(2, 8)}
        },
        128: {
            'in_channels': [item * ch for item in [1, 2, 4, 8, 16]],
            'out_channels': [item * ch for item in [2, 4, 8, 16, 16]],
            'downsample': [True] * 5 + [False],
            'resolution': [64, 32, 16, 8, 4, 4],
            'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')] for i in range(2, 8)}
        },
        64: {
            'in_channels': [item * ch for item in [1, 2, 4, 8]],
            'out_channels': [item * ch for item in [2, 4, 8, 16]],
            'downsample': [True] * 4 + [False],
            'resolution': [32, 16, 8, 4, 4],
             'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')] for i in range(2, 7)}
        },
        32: {
            'in_channels': [item * ch for item in [4, 4, 4]],
            'out_channels': [item * ch for item in [4, 4, 4]],
            'downsample': [True, True, False, False],
            'resolution': [16, 16, 16, 16],
            'attention': {2 ** i: 2 ** i in [int(item) for item in attention.split('_')] for i in range(2, 6)}
        }
    }
    return arch


# 判别器
class Discriminator(nn.Module):
    def __init__(self, D_ch=64, D_depth=2, resolution=128, D_kernel_size=3, D_attn='64', n_classes=1000, num_D_SVs=1,
                 num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False), D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
                 SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False, D_init='ortho', skip_init=False,
                 D_param='SN'):
        super(Discriminator, self).__init__()
        # 通道宽度单位
        self.ch = D_ch
        # 每阶段残差块数量
        self.D_depth = D_depth
        # 输入分辨率
        self.resolution = resolution
        # 卷积核大小
        self.kernel_size = D_kernel_size
        # 注意力机制
        self.attention = D_attn
        # 类别数
        self.n_classes = n_classes
        # 激活函数
        self.activation = D_activation
        # 初始化风格
        self.init = D_init
        # 参数化风格
        self.D_param = D_param
        # 谱归一化的epsilon
        self.SN_eps = SN_eps
        # 使用fp16
        self.fp16 = D_fp16
        # 模型结构字典
        self.arch = D_arch(self.ch, self.attention)[resolution]
        # 参数计数
        self.param_count = 0
        # 卷积和归一化类型选择
        if self.D_param == 'SN':
            self.which_conv = functools.partial(layers.SNConv2d, kernel_size=3, padding=1,
                                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_linear = functools.partial(layers.SNLinear, num_svs=num_D_SVs,
                                                  num_itrs=num_D_SV_itrs, eps=self.SN_eps)
            self.which_embedding = functools.partial(layers.SNEmbedding, num_svs=num_D_SVs,
                                                     num_itrs=num_D_SV_itrs, eps=self.SN_eps)
        # 输入层
        self.input_conv = self.which_conv(3, self.arch['in_channels'][0])
        # 构建后续模型结构
        self.blocks = []
        for index in range(len(self.arch['out_channels'])):
            self.blocks += [[
                DBlock(in_channels=self.arch['in_channels'][index]
                if d_index == 0 else self.arch['out_channels'][index],
                out_channels=self.arch['out_channels'][index], which_conv=self.which_conv, activation=self.activation,
                pre_activation=True,
                down_sample=(nn.AvgPool2d(2) if self.arch['downsample'][index] and d_index == 0 else None))
                for d_index in range(self.D_depth)
            ]]
            # 使用注意力模块
            if self.arch['attention'][self.arch['resolution'][index]]:
                self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]
        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        # 输出层
        self.linear = self.which_linear(self.arch['out_channels'][-1], output_dim)
        # 嵌入层
        self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
        # 初始化权重
        if not skip_init:
            self.init_weights()
        # 设置优化器
        self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
        if D_mixed_precision:
            self.optim = Adam16(params=self.parameters(), lr=self.lr,
                                betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
        else:
            self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                                    betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized.')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])

    def forward(self, x, y=None):
        h = self.input_conv(x)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        h = torch.sum(self.activation(h), [2, 3])
        out = self.linear(h)
        out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
        return out


# 组合判别器和生成器
class BigGAN(nn.Module):
    def __init__(self, G, D):
        super(BigGAN, self).__init__()
        self.G = G
        self.D = D

    def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False, split_D=False):
        with torch.set_grad_enabled(train_G):
            G_z = self.G(z, self.G.shared(gy))
            if self.G.fp16 and not self.D.fp16:
                G_z = G_z.float()
            if self.D.fp16 and not self.G.fp16:
                G_z = G_z.half()
        if split_D:
            D_fake = self.D(G_z, gy)
            if x is not None:
                D_real = self.D(x, dy)
                return D_fake, D_real
            else:
                if return_G_z:
                    return D_fake, G_z
                else:
                    return D_fake
        else:
            D_input = torch.cat([G_z, x], 0) if x is not None else G_z
            D_class = torch.cat([gy, dy], 0) if dy is not None else gy
            D_out = self.D(D_input, D_class)
            if x is not None:
                return torch.split(D_out, [G_z.shape[0], x.shape[0]])
            else:
                if return_G_z:
                    return D_out, G_z
                else:
                    return D_out
