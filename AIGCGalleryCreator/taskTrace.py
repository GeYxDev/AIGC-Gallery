import argparse
from enum import Enum
from typing import Optional, Union, List

import torch

import manageQueue
from ImageCreator import VQGANImageCreator, BigGANImageCreator
from VideoCreator import LXTVideoCreator, CogVideoCreator


class TaskStatus(Enum):
    """
    任务控制块的状态
    """
    WAITING = 0  # 处于等待队列
    RUNNING = 1  # 正在运行
    SUCCESS = 2  # 成功运行完毕
    INTERRUPT = 3  # 运行中断
    ERROR = 4  # 运行错误


class TCB:
    """
    基础任务控制块
    """
    def __init__(self, taskId: str, prompt: str):
        self.taskId = taskId
        self.prompt = prompt
        self.taskStatus = TaskStatus.WAITING

    def __eq__(self, other):
        if isinstance(other, TCB):
            return self.taskId == other.taskId
        elif isinstance(other, str):
            return self.taskId == other
        return False

    def __str__(self):
        return f"taskId <{self.taskId}> "


class ImageTCB(TCB):
    """
    图像生成任务控制块
    """
    def __init__(self, taskId: str, prompt: Optional[Union[str, List[str]]], modelType: str, iteration: int):
        super().__init__(taskId, prompt)
        self.modelType = modelType
        self.iteration = iteration
        self.inspectRecord = None


class VideoTCB(TCB):
    """
    视频生成任务控制块
    """
    def __init__(self, taskId: str, prompt: str, modelType: str, iteration: int, duration: int):
        super().__init__(taskId, prompt)
        self.modelType = modelType
        self.iteration = iteration
        self.duration = duration


class Executor:
    """
    生成任务执行控制器
    """
    def __init__(self, tcb: TCB, finishedQueue: manageQueue, exceptionQueue: manageQueue, queryTime: float):
        self.tcb = tcb
        self.finishedQueue = finishedQueue
        self.exceptionQueue = exceptionQueue
        self.queryTime = queryTime
        if isinstance(self.tcb, ImageTCB):
            # 初始化图像生成任务
            if tcb.modelType == "VQGAN":
                self.creator = VQGANImageCreator(self.constructVQGANImageCreator())
            elif tcb.modelType == "BigGAN":
                self.creator = BigGANImageCreator(self.constructBigGANImageCreator())
            else:
                raise Exception("Unknown model type.")
        elif isinstance(self.tcb, VideoTCB):
            # 初始化视频生成任务
            if tcb.modelType == "LTX-Video":
                self.creator = LXTVideoCreator(self.constructLTXVideoCreator())
            elif tcb.modelType == "CogVideo":
                self.creator = CogVideoCreator(self.constructCogVideoCreator())
            else:
                raise Exception("Unknown model type.")
        else:
            raise Exception("Unknown type generation task.")
        # 生成任务执行状态
        self.generateStatus = {
            'generated': False,  # 生成任务完成或终止
            'interrupt': False,  # 终止生成任务
            'finish': False,  # 线程任务是否结束
            'currentIter': 0  # 已记录迭代次数
        }

    # 执行生成任务
    def run(self):
        # noinspection PyBroadException
        try:
            self.creator.train(self.generateStatus)
            if self.generateStatus['interrupt']:
                # 若中断生成任务
                self.tcb.taskStatus = TaskStatus.INTERRUPT
                self.exceptionQueue.push(self.tcb)
                print('Image generation interrupted.')
            else:
                # 将已完成生成任务的控制块放入作品生成完成队列中
                self.generateStatus['generated'] = True
                self.finishedQueue.push(self.tcb)
                self.tcb.taskStatus = TaskStatus.SUCCESS
        except:
            # 生成任务执行过程出现异常
            self.generateStatus['interrupt'] = True
            self.tcb.taskStatus = TaskStatus.ERROR
            self.exceptionQueue.push(self.tcb)
        finally:
            self.generateStatus['finish'] = True

    # 中断生成任务
    def stop(self):
        self.generateStatus['interrupt'] = True

    # 获得迭代次数
    def getCurIter(self):
        return self.creator.getCurIter()

    # 构建VQGAN-CLIP图像生成参数
    def constructVQGANImageCreator(self):
        if not isinstance(self.tcb, ImageTCB):
            raise TypeError("Wrong construction.")
        args = argparse.Namespace(
            prompts=self.tcb.prompt,
            max_iterations=self.tcb.iteration,
            save_every=20,
            clip_model='ViT-B/32',
            vqgan_config='./checkpoints/vqgan_imagenet_f16_16384.yaml',
            vqgan_checkpoint='./checkpoints/vqgan_imagenet_f16_16384.ckpt',
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            learning_rate=0.1,
            cutn=64,
            cut_pow=1.,
            output=f'./transfer/{self.tcb.taskId}'
        )
        return args

    # 构建BigGAN-CLIP图像生成参数
    def constructBigGANImageCreator(self):
        if not isinstance(self.tcb, ImageTCB):
            raise TypeError("Wrong construction.")
        args = argparse.Namespace(
            prompts=self.tcb.prompt,
            max_iterations=self.tcb.iteration,
            save_every=20,
            clip_model='ViT-B/32',
            learning_rate=0.1,
            cutn=64,
            truncation=1,
            class_ent_reg=0.0001,
            output=f'./transfer/{self.tcb.taskId}'
        )
        return args

    # 构建LTXVideo视频生成参数
    def constructLTXVideoCreator(self):
        if not isinstance(self.tcb, VideoTCB):
            raise TypeError("Wrong construction.")
        args = argparse.Namespace(
            prompts=self.tcb.prompt,
            max_iterations=self.tcb.iteration,
            model_checkpoint='./checkpoints/LTX-Video-2b',
            width=704,
            height=480,
            num_frames=self.tcb.duration * 24 + 1,
            guidance_scale=3.0,
            dtype=torch.bfloat16,
            seed=42,
            fps=24,
            output=f'./transfer/{self.tcb.taskId}'
        )
        return args

    # 构建CogVideo视频生成参数
    def constructCogVideoCreator(self):
        if not isinstance(self.tcb, VideoTCB):
            raise TypeError("Wrong construction.")
        args = argparse.Namespace(
            prompts=self.tcb.prompt,
            max_iterations=self.tcb.iteration,
            model_checkpoint='./checkpoints/CogVideoX-5b',
            width=1360,
            height=768,
            num_frames=self.tcb.duration * 16 + 1,
            guidance_scale=6.0,
            dtype=torch.bfloat16,
            seed=42,
            fps=16,
            output=f'./transfer/{self.tcb.taskId}'
        )
        return args
