import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import wget
from huggingface_hub import snapshot_download

# 下载LTX-Video模型预训练文件
snapshot_download("Lightricks/LTX-Video", local_dir='./LTX-Video-2b', local_dir_use_symlinks=False, repo_type='model')

# 下载CogVideo模型预训练文件
snapshot_download("THUDM/CogVideoX-5b", local_dir='./CogVideoX-5b', local_dir_use_symlinks=False, repo_type='model')

# 下载VQGAN模型预训练文件
wget.download('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1')
wget.download('https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1')

# 下载ChineseCLIP模型预训练文件
wget.download('https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt')
