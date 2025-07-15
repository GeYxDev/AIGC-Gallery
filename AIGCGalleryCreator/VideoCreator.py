import os
from datetime import datetime

import torch

from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler, LTXPipeline
from diffusers.utils import export_to_video

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# LXTVideo视频生成
class LXTVideoCreator:
    def __init__(self, args):
        self.args = args
        # 加载LXTVideo预训练模型
        self.pipe = LTXPipeline.from_pretrained(args.model_checkpoint, torch_dtype=args.dtype).to(device)

    # 视频生成
    def train(self, generateStatus):
        try:
            # 视频生成
            video_generate = self.pipe(
                generateStatus=generateStatus,
                width=self.args.width,
                height=self.args.height,
                prompt=self.args.prompts,
                num_inference_steps=self.args.max_iterations,
                num_frames=self.args.num_frames,
                guidance_scale=self.args.guidance_scale,
                generator=torch.Generator().manual_seed(self.args.seed)
            ).frames[0]
            # 保存生成的视频
            export_to_video(video_generate, os.path.join(self.args.output, f'{datetime.now().timestamp() * 1000}.mp4'), fps=self.args.fps)
        except KeyboardInterrupt:
            pass

    # 获得当前迭代次数
    def getCurIter(self):
        return self.pipe.cur_iter


# CogVideo视频生成
class CogVideoCreator:
    def __init__(self, args):
        self.args = args
        # 加载CogVideo预训练模型
        self.pipe = CogVideoXPipeline.from_pretrained(args.model_checkpoint, torch_dtype=args.dtype)
        # 设置调度器
        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config, timestep_spacing="trailing")
        # 优化设置
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.to(device)

    # 视频生成
    def train(self, generateStatus):
        try:
            # 视频生成
            video_generate = self.pipe(
                generateStatus=generateStatus,
                width=self.args.width,
                height=self.args.height,
                prompt=self.args.prompts,
                num_inference_steps=self.args.max_iterations,
                num_frames=self.args.num_frames,
                use_dynamic_cfg=True,
                guidance_scale=self.args.guidance_scale,
                generator=torch.Generator().manual_seed(self.args.seed)
            ).frames[0]
            # 保存生成的视频
            export_to_video(video_generate, os.path.join(self.args.output, f'{datetime.now().timestamp() * 1000}.mp4'), fps=self.args.fps)
        except KeyboardInterrupt:
            pass

    # 获得当前迭代次数
    def getCurIter(self):
        return self.pipe.cur_iter
