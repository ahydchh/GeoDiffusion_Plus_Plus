import os
import cv2
import numpy as np
from PIL import Image

from argparse import ArgumentParser
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import torch

from mmcv import Config
from mmdet.datasets import build_dataset
from utils.data.nuimage import NuImageDataset
from utils.data.coco_stuff import COCOStuffDataset

########################
# Set random seed
#########################
from accelerate.utils import set_seed
set_seed(0)

########################
# Parsers
#########################
parser = ArgumentParser(description='Generation script')
parser.add_argument('ckpt_path', type=str)
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
ckpt_path = args.ckpt_path
pipe = StableDiffusionPipeline.from_pretrained(ckpt_path, torch_dtype=torch.float16)  
if args.use_dpmsolver:
  assert '0.16.0' in diffusers.__version__, "Be default, we adopt diffusers==0.16.0 to adopt DPMSolver++ for inference on COCO-Stuff."
  from diffusers import DPMSolverMultistepScheduler
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")


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


########################
# Set index range
#########################
prompts = [dataset[i]['text'] for i in range(len(dataset))]
########################
# Generation
#########################
scale = args.cfg_scale
n_samples = args.nsamples

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = True

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety


dpm_flag = "_dpmsolver" if args.use_dpmsolver else ""
if len(ckpt_path.split("/"))==1:
  root = os.path.join('work_dirs/output_img', f"{ckpt_path}", '{}/{}-{}_scale{}_samples{}{}'.format(args.split, args.split, args.num_inference_steps, str(scale), str(n_samples), dpm_flag))  
else:
  root = os.path.join('work_dirs/output_img', f"{ckpt_path.split('/')[-2]}_{ckpt_path.split('/')[-1]}", '{}/{}-{}_scale{}_samples{}{}'.format(args.split, args.split, args.num_inference_steps, str(scale), str(n_samples), dpm_flag))  
for idx in range(n_samples):
  group_dir = os.path.join(root, f"group_{idx}")
  if not os.path.exists(group_dir):
    os.makedirs(group_dir)

for i, prompt in enumerate(prompts):
  # make directory
  path = dataset.data_infos[i]["file_name"]
  img_height = dataset.data_infos[i]["height"]
  img_width = dataset.data_infos[i]["width"]
  # os.makedirs(os.path.join(root, os.path.dirname(path)), exist_ok=True)
  
  pass_ = True
  for idx in range(5):
    group_dir = os.path.join(root, f"group_{idx}")
    if not os.path.exists(os.path.join(group_dir, path)):
      pass_ = False
  
  if pass_ :
    print(f"passed image {i}")
  else:
    with torch.autocast("cuda"):
      images = pipe(n_samples*[prompt], guidance_scale=scale, num_inference_steps=args.num_inference_steps, height=int(height), width=int(width)).images
    
    # save results
    for idx, image in enumerate(images):
        image = np.asarray(image)
        image = Image.fromarray(image, mode='RGB')
        image = image.resize((img_width, img_height))
        group_dir = os.path.join(root, f"group_{idx}")
        image.save(os.path.join(group_dir, path))