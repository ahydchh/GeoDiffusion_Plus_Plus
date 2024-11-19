import os
import cv2
import numpy as np
from PIL import Image
import time
from argparse import ArgumentParser
from accelerate import Accelerator
from accelerate import PartialState
from diffusers import StableDiffusionPipeline
import torch
import shutil
from mmcv import Config
from mmdet.datasets import build_dataset
from utils.data.nuimage import NuImageDataset
from utils.data.coco_stuff import COCOStuffDataset

########################
# Set random seed
#########################

########################
# Parsers
#########################
parser = ArgumentParser(description='Generation script')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--nsamples', type=int, default=5)
parser.add_argument('--cfg_scale', type=float, default=5)
parser.add_argument("--use_dpmsolver", action="store_true")
parser.add_argument('--num_inference_steps', type=int, default=100)

# copy from training script
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default=None,
    help="The config of the Dataset, leave as None if there's only one config.",
)

parser.add_argument(
    "--prompt_version",
    type=str,
    default="v1",
    help="Text prompt version. Default to be version3 which is constructed with only camera variables",
)

parser.add_argument(
    "--num_bucket_per_side",
    type=int,
    default=None,
    nargs="+", 
    help="Location bucket number along each side (i.e., total bucket number = num_bucket_per_side * num_bucket_per_side) ",
)

args = parser.parse_known_args()[0]

print("{}".format(args).replace(', ', ',\n'))

########################
# Build pipeline
#########################

########################
# Load dataset
# Note: remember to disable randomness in data augmentations !!!!!! (TODO CHECK)
#########################
if (len(args.num_bucket_per_side) == 1):
    args.num_bucket_per_side *= 2
dataset_cfg = Config.fromfile(args.dataset_config_name)
dataset_cfg.data.train.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=args.num_bucket_per_side))
dataset_cfg.data.val.update(dict(prompt_version=args.prompt_version, num_bucket_per_side=args.num_bucket_per_side))
dataset_cfg.data.train.pipeline[3]["flip_ratio"] = 0.0
dataset_cfg.data.val.pipeline[3]["flip_ratio"] = 0.0

width, height = dataset_cfg.data.train.pipeline[2].img_scale if args.split == 'train' else dataset_cfg.data.val.pipeline[2].img_scale
dataset = build_dataset(dataset_cfg.data.train) if args.split == 'train' else build_dataset(dataset_cfg.data.val)

print('Image resolution: {} x {}'.format(width, height))
print(len(dataset))
import pdb
pdb.set_trace()
from tqdm import trange
for i in trange(len(dataset)):
    source_file = dataset[i]["filename"]
    destination_file = os.path.join("data/eval_nuimages","/".join(source_file.split("/")[3:]))
    # 获取文件的目录路径
    folder_path = os.path.dirname(destination_file)

    # 创建文件夹（包括任何中间缺少的目录）
    os.makedirs(folder_path, exist_ok=True)
    # 复制文件
    shutil.copy(source_file, destination_file)