import argparse
import logging
import math
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import itertools
from pathlib import Path
from typing import Iterable, Optional

import gc
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from mmcv import Config
from mmdet.datasets import build_dataset
from utils.data.nuimage import NuImageDataset
from utils.data.coco_stuff import COCOStuffDataset
import utils.misc as misc


import torchvision
from torchvision.transforms.functional import normalize
from detr_od.models.losses import TaskAlignedFocalLoss
from mmdet.models.losses import FocalLoss, L1Loss, GIoULoss
from mmcv import Config
from mmdet.apis import inference_detector
from detr_ssod.apis.inference import init_detector, save_result
from detr_ssod.utils import patch_config
from mmdet.core import bbox_xyxy_to_cxcywh

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from mmcv.parallel import DataContainer as DC
from PIL import Image

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # model pre-trained name (download from huggingface) or path (local) 
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
    
    parser.add_argument("--bucket_sincos_embed", action="store_true", help="Whether to use 2D sine-cosine embedding for bucket locations")
    
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    
    parser.add_argument(
        "--train_text_encoder_params", 
        type=str, 
        default=["token_embedding", "position", "encoder", "final_layer_norm"], 
        nargs="+", 
        help="token_embedding, position (position_embedding & position_ids), encoder, final_layer_norm, added_embedding (means tuning added tokens while fixing word tokens) "
    )
    
    # foreground loss specifics
    parser.add_argument("--foreground_loss_mode", type=str, default=None, help="None, constant and area")
    
    parser.add_argument("--foreground_loss_weight", type=float, default=1.0, help="Might be utilized differently with respect to loss mode")
    
    parser.add_argument("--foreground_loss_norm", action="store_true", help="Whether to normalize bbox mask")
    
    parser.add_argument("--feat_size", type=int, default=64, help="Feature size of LDMs. Default to be 64 for stable diffusion v1.5")
    
    # unconditional generation specifics
    parser.add_argument("--uncond_prob", type=float, default=0.0, help="Probability to downgrade to unconditional generation")
    
    # data specifics
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    
    # output specifics
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    
    # data augmentation specifics
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    # optimization specifics
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_text_ratio",
        type=float,
        default=1.0,
        help="Ratio of text encoder LR with respect to UNet LR",
    )
    parser.add_argument(
        "--lr_text_layer_decay",
        type=float,
        default=1.0,
        help="Layer-wise LR decay ratio of text encoder",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--use_ema_text", action="store_true", help="Whether to use EMA model for text encoder.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    # huggingface specifics
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    
    # logging specifics
    # logging_dir = os.path.join(args.output_dir, args.logging_dir)
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # evaluation specifics
    parser.add_argument("--save_ckpt_freq", type=int, default=10000)
    
    parser.add_argument(
        "--name",
        type=str,
        default="checkpoint",
    )
    parser.add_argument(
        "--reward_checkpoint",
        type=str,
        default="../Semi-DETR/models/semi_detr_coco_full.pth",
    )
    parser.add_argument(
        "--min_timestep_rewarding",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_timestep_rewarding",
        type=int,
        default=200,
    )
    
    parser.add_argument(
        "--grad_scale", type=float, default=0.1, help="Scale divided for grad loss value."
    )
    parser.add_argument(
        "--loss_cls_scale", type=float, default=1.0
    )
    parser.add_argument("--reward_config", help="Reward model config file")
    parser.add_argument(
        "--timestep_sampling_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--timestep_sampling_end",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--score_thr",
        type=float,
        default=0.3,
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.

        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.

        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]
    # From CompVis LitEMA implementation
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

        del self.collected_params
        gc.collect()


def check_existence(text, candidates):
    for each in candidates:
        if each in text:
            return True
    return False

class SplitEmbedding(nn.Module):
    def __init__(self, embed, num_fixed):
        """
        Args:
            embed (nn.Embedding)
            num_fixed: default to be the first num_fixed tokens
        """
        super(SplitEmbedding, self).__init__()
        self.embed_origin = embed
        self.padding_idx = embed.padding_idx
        self.max_norm = embed.max_norm
        self.norm_type = embed.norm_type
        self.scale_grad_by_freq = embed.scale_grad_by_freq
        self.sparse = embed.sparse
        
        self.num_embeddings, self.embedding_dim = embed.weight.shape
        self.num_fixed = num_fixed
        self.num_tuned = self.num_embeddings - num_fixed
        print("Spliting original embedding with shape ({}, {}) to ({}, {}) and ({}, {})".format(self.num_embeddings, self.embedding_dim, self.num_fixed, self.embedding_dim, self.num_tuned, self.embedding_dim))
    
        self.fixed_tokens = nn.Parameter(torch.zeros(self.num_fixed, self.embedding_dim), requires_grad=False)
        self.tuned_tokens = nn.Parameter(torch.zeros(self.num_tuned, self.embedding_dim))
        self.fixed_tokens.data.copy_(embed.weight.data[:num_fixed])
        self.tuned_tokens.data.copy_(embed.weight.data[num_fixed:])
        
    def forward(self, input):
        weight = torch.cat([self.fixed_tokens, self.tuned_tokens], dim=0)
        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

def test_pipeline(img):
    assert isinstance(img, torch.Tensor)
    # if not isinstance(img, torch.Tensor):
    #     # Load
    #     client = mmcv.FileClient(backend='disk')
    #     img_bytes = client.get(img)
    #     numpy_img = mmcv.imfrombytes(
    #             img_bytes, flag='color', channel_order='bgr')
    #     ori_img = torch.tensor(numpy_img,requires_grad=True,dtype=torch.float32)
    # else:
    ori_img = img
    numpy_img = img.detach().cpu().numpy()
    # print(numpy_img)
    # image = Image.fromarray(numpy_img, mode='RGB')
    # image.save(os.path.join(f'/gpfs/essfs/iat/Tsinghua/chihh/Semi-DETR/test_dir/out_start.jpg'))

    results = {}
    results['filename'] = None
    results['ori_filename'] = None
    results['img'] = ori_img
    results['img_shape'] = ori_img.shape # (427, 640, 3)
    results['ori_shape'] = ori_img.shape
    results['img_fields'] = ['img']
    
    # MultiScaleFlipAug: img_scale=(1333, 800) flip=False
    results['scale'] = (1333, 800) # (w, h)
    results['flip']  = False
    results['flip_direction'] = None
    
    results['img'] = results['img'].permute(2,0,1).contiguous()
    
    # Resize: keep_ratio=True
    # Resize the image while keeping the aspect ratio
    h, w = ori_img.shape[:2]
    max_long_edge = max(results['scale'])
    max_short_edge = min(results['scale'])
    scale_factor = min(max_long_edge / max(h, w),
                        max_short_edge / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)

    img = F.resize(results['img'], (new_size[1], new_size[0]), interpolation=InterpolationMode.BILINEAR)
    
    # Get the new height and width
    new_h, new_w = img.shape[-2:]  # Torch Tensor shape is (C, H, W) or (H, W)
    h, w = results['img'].shape[-2:]
    # Calculate scale factors
    w_scale = new_w / w
    h_scale = new_h / h
    # Update the results
    results['img'] = img #(3, 1199, 800)
    # Create the scale factor as a torch.Tensor
    scale_factor = torch.tensor([w_scale, h_scale, w_scale, h_scale], dtype=torch.float32)
    # Update other properties in the results
    results['img_shape'] = (img.shape[1],img.shape[2],img.shape[0])  # Torch Tensor shape
    results['pad_shape'] = img.shape  # Same as img_shape in case no padding
    results['scale_factor'] = scale_factor
    results['keep_ratio'] = True
    # RandomFlip: do nothing
    
    # Normalize: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]
    
    # to rgb:
    results['img'] = results['img'][[2, 1, 0], :, :].to(torch.float64)  # Swap channels if needed (assuming tensor shape is C, H, W)
    # Normalize the tensor (subtract mean and divide by std)
    results['img'] = F.normalize(results['img'],mean,std)
    results['img_norm_cfg'] = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    
    # Pad: size_divisor=1 seems do nothing
    results['pad_shape'] = (results['img'].shape[1],results['img'].shape[2],results['img'].shape[0])
    results['pad_fixed_size'] = None
    results['pad_size_divisor'] = 1
    
    # DefaultFormatBundle: img_to_float=True (here the dtype shall be float32)
    results['img'] = results['img'].to(torch.float32)
    results['img'] = DC(results['img'], padding_value=0, stack=True)
    # Collect 
    data={}
    img_meta = {}
    for key in ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg']:
        img_meta[key] = results[key]
    data['img_metas']=DC(img_meta,cpu_only=True)
    data['img']=results['img']
    data = {'img':[data['img']], 'img_metas':[data['img_metas']]}
    return numpy_img, data, ori_img

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    # accelerate deals with fp16, accumulation and tensorboard logging
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if accelerator.is_local_main_process:
        print("{}".format(args).replace(', ', ',\n'))
    
    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if "token_embedding" in args.train_text_encoder_params and "added_embedding" in args.train_text_encoder_params:
        raise ValueError(
            "Added_embedding suggests only tuning added tokens while fixing existing word tokens, which contradicts against token_embedding. "
        )

    if not args.train_text_encoder and args.use_ema_text:
        raise ValueError(
            "Use EMA model for text encoder only when we train text encoder. "
        )

    # If passed along, set the training seed now.
    if args.seed is not None:
        # set_seed(args.seed)
        print('Set random seed to: {}'.format(args.seed))
        seed = args.seed + accelerator.process_index
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_local_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load models and create wrapper for stable diffusion
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    num_words = text_encoder.get_input_embeddings().num_embeddings
    # add location tokens
    if args.num_bucket_per_side is not None:
        if len(args.num_bucket_per_side) == 1:
            args.num_bucket_per_side *= 2
        new_tokens = ["<l{}>".format(i) for i in range(int(args.num_bucket_per_side[0] * args.num_bucket_per_side[1]))]
        tokenizer.add_tokens(new_tokens)
        
        # expand text_encoder nn.Embedding.weight if needed
        num_vocabs = len(tokenizer.get_vocab())
        num_embeds, dim = text_encoder.get_input_embeddings().weight.size()
        if num_vocabs != num_embeds:
            print('Expanding text encoder embedding size from {} to {}'.format(num_embeds, num_vocabs))
            text_encoder.resize_token_embeddings(num_vocabs)
            if args.bucket_sincos_embed:
                print('Use 2D sine-cosine embeddings to initialize bucket location tokens')
                from utils.misc import get_2d_sincos_pos_embed
                pos_embed = get_2d_sincos_pos_embed(dim, args.num_bucket_per_side)
                pos_embed = torch.from_numpy(pos_embed).float()
                # pos_embed = pos_embed / torch.norm(pos_embed, p=2, dim=1, keepdim=True) * 0.3854
                text_encoder.text_model.embeddings.token_embedding.weight.data[num_embeds:] = pos_embed
        
        assert len(tokenizer.get_vocab()) == text_encoder.get_input_embeddings().weight.size()[0]
    
    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)
    else:
        for name, param in text_encoder.named_parameters():
            # if 'bias' not in name or 'norm' not in name:
            if not check_existence(name, args.train_text_encoder_params):
                param.requires_grad_(False)
        if "added_embedding" in args.train_text_encoder_params:
            text_encoder.text_model.embeddings.token_embedding = SplitEmbedding(text_encoder.text_model.embeddings.token_embedding, num_words)
        
        tuned_params = [name for name, param in text_encoder.named_parameters() if param.requires_grad]
        print('Number of tuned parmeters: {}'.format(len(tuned_params)))
        assert len(tuned_params) > 0, "No text encoder parameters are being tuned! Set train_text_encoder to be false!"
        # if accelerator.is_local_main_process:
        #     print("Text encoder tuned params include: {}".format(',\n'.join(tuned_params)))

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # If used, actual lr = base lr * accum * bs * num_process (base lr corresponds to a single data sample - not the usual 256 based)
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
        
    # Layer-wise Learning Rate Decay
    params_to_optimize = misc.param_groups_lrd(
        text_encoder,
        weight_decay=args.adam_weight_decay,
        base_lr=args.learning_rate * args.lr_text_ratio,
        layer_decay=args.lr_text_layer_decay,
        verbose=accelerator.is_local_main_process
    ) + [{'params': unet.parameters(), 'name': 'unet'}] if args.train_text_encoder else unet.parameters()
    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # if accelerator.is_local_main_process:
    #     print(optimizer)

    # MODIFY: to be consistent with the pre-trained model
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for example in examples:
            caption = example[caption_column]
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        # pad in the collate_fn function
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids

    dataset_args = dict(
        prompt_version=args.prompt_version, 
        num_bucket_per_side=args.num_bucket_per_side,
        foreground_loss_mode=args.foreground_loss_mode, 
        foreground_loss_weight=args.foreground_loss_weight,
        foreground_loss_norm=args.foreground_loss_norm,
        feat_size=args.feat_size,
    )
    dataset_args_train = dict(
        uncond_prob=args.uncond_prob,
    )
    dataset_cfg = Config.fromfile(args.dataset_config_name)
    dataset_cfg.data.train.update(**dataset_args)
    dataset_cfg.data.train.update(**dataset_args_train)
    train_dataset = build_dataset(dataset_cfg.data.train)
    
    ##################################
    # Prepare dataloader
    ##################################
    def collate_fn(examples):
        filenames = [example["filename"] for example in examples]
        prompts = [example["text"] for example in examples]
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        idxs = [example["idx"] for example in examples]
        input_ids = tokenize_captions(examples)
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt") # pad to the longest of current batch (might differ between cards)
        bbox_mask = torch.stack([example["bbox_mask"] for example in examples]).unsqueeze(1).to(memory_format=torch.contiguous_format).float() if args.foreground_loss_mode else None # [B, 1, H, W]
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
            "bbox_mask": bbox_mask,
            "idxs": idxs,
            "filenames": filenames,
            "prompts": prompts
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # convert to the corresponding distributed version one by one for any kinds of objects
    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader
        )
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader
        )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters())
    if args.use_ema_text:
        ema_text_encoder = EMAModel(text_encoder.parameters())

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_local_main_process:
        saved_args = copy.copy(args)
        saved_args.num_bucket_per_side = ' '.join([str(each) for each in saved_args.num_bucket_per_side])
        saved_args.train_text_encoder_params = ' '.join(saved_args.train_text_encoder_params)
        accelerator.init_trackers("text2image-fine-tune", config=vars(saved_args))

    cfg = Config.fromfile(args.reward_config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    reward_model = init_detector(cfg, args.reward_checkpoint, device=accelerator.device)
    if 'nuimage' in args.reward_config:
        rwd_loss_func = reward_model.bbox_head.rwd_loss_single
    else:
        rwd_loss_func = reward_model.model().bbox_head.loss_single
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(args.name)
    global_step = 0

    for epoch in range(args.num_train_epochs):
        loss_per_epoch = 0.
        pretrain_loss_per_epoch = 0.
        reward_loss_per_epoch = 0.
        train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.
        
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # for gradient accumulation
            selected_idx = 0
            with accelerator.accumulate(unet):
                # Convert images to latent space
                # images: [B, 3, 512, 512], latents: [B, 4, 64, 64]
                import pdb
                pdb.set_trace()
                print("===============",batch["filenames"][selected_idx], batch["prompts"][selected_idx])
                # latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = vae.encode(torch.stack([batch["pixel_values"][selected_idx] for _ in range(8)]).to(weight_dtype)).latent_dist.sample()
                
                # multiply with the scalr factor
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # print(" === noise_scheduler.num_train_timesteps:",noise_scheduler.num_train_timesteps)
                # timesteps = torch.randint(args.timestep_sampling_start, args.timestep_sampling_end, (bsz,), device=latents.device)
                
                timesteps = torch.tensor([50,100,150,200,250,300,350,400], device=latents.device)
                # timesteps = torch.tensor([450,500,550,600,650,700,750,800], device=latents.device)
                
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # Return last_hidden_state of the text: [B, L, D=768]
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                encoder_hidden_states = text_encoder(torch.stack([batch["input_ids"][selected_idx] for _ in range(8)]))[0]
                
                # Predict the noise residual and compute loss
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                """
                Rewarding start
                """
                # Predict the single-step denoised latents
                pred_original_sample = [
                    noise_scheduler.step(noise, t, noisy_latent).pred_original_sample.to(weight_dtype) \
                        for (noise, t, noisy_latent) in zip(noise_pred, timesteps, noisy_latents)
                ]
                pred_original_sample = torch.stack(pred_original_sample)
                
                # Map the denoised latents into RGB images
                # scalr factor is 0.18215
                # pred_original_sample = 1 / vae.config.scaling_factor * pred_original_sample
                
                pred_original_sample = 1 / 0.18215 * pred_original_sample
                image = vae.decode(pred_original_sample.to(weight_dtype)).sample

                # (bs, c, h, w) -> (bs, h, w, c) rgb -> bgr
                # that's because the input of test_pipeline is numpy like tensor with (bs, h, w, c) and bgr format
                # print("======")
                # print(batch["pixel_values"].shape,image.shape)
                # gt_np = [img.permute(1, 2, 0).cpu().numpy() for img in batch["pixel_values"]]
                # im_np = [img.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32) for img in image]
                
                timestep_mask = (args.min_timestep_rewarding <= timesteps.reshape(-1, 1)) & (timesteps.reshape(-1, 1) <= args.max_timestep_rewarding)
                
                """
                Compute reward loss (semi-detr)
                """
                
                mean = torch.tensor([127.5, 127.5, 127.5]).to(batch["pixel_values"][selected_idx].device) 
                std = torch.tensor([127.5, 127.5, 127.5]).to(batch["pixel_values"][selected_idx].device) 
                mean = mean.view(3, 1, 1)  # 图像是 [C, H, W] 的张量
                std = std.view(3, 1, 1)
                
                processed_gt = [test_pipeline((img * std + mean).permute(1,2,0)[...,[2,1,0]]) for img in batch["pixel_values"]]
                processed_out = [test_pipeline((img * std + mean).permute(1,2,0)[...,[2,1,0]]) for img in image]
                ground_truth = [data[1] for data in processed_gt]
                prediction = [data[1] for data in processed_out]
                
                test_dir = "test_dir"
                bbox_outputs, outputs, output_metas = inference_detector(reward_model, prediction, tensor=True)
                bbox_labels, labels, label_metas = inference_detector(reward_model, ground_truth, tensor=True)
                img_height = train_dataset.data_infos[batch['idxs'][selected_idx]]["height"]
                img_width = train_dataset.data_infos[batch['idxs'][selected_idx]]["width"]
                test_gt_image = Image.fromarray(processed_gt[0][0][:,:,::-1].astype(np.int8), mode='RGB')
                test_gt_image = test_gt_image.resize((img_width, img_height))
                test_gt_image.save(os.path.join(test_dir,f"gt_{batch['idxs'][selected_idx]}.jpg"))
                test_out_image = Image.fromarray(processed_out[0][0][:,:,::-1].astype(np.int8), mode='RGB')
                test_out_image = test_out_image.resize((img_width, img_height))
                test_out_image.save(os.path.join(test_dir,f"out_{batch['idxs'][selected_idx]}.jpg"))
                # test_gt_image.save(os.path.join(f'test_dir/gt_{batch["idxs"][selected_idx]}.jpg'))
                for score_thr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
                    score_path = os.path.join(test_dir,str(score_thr))
                    os.makedirs(score_path,exist_ok=True)
                    save_result(
                            reward_model, processed_gt[0][0], bbox_labels[0], score_thr=score_thr, out_file=os.path.join(score_path, f'fin_gt_{batch["idxs"][selected_idx]}.jpg')
                        )
                    for test_idx in range(8):
                        test_out_image = processed_out[test_idx][0]
                        test_out_image = Image.fromarray(test_out_image.astype(np.int8), mode='RGB')
                        test_out_image = test_out_image.resize((img_width, img_height))
                        # test_out_image.save(os.path.join(f'test_dir/out_{batch["idxs"][selected_idx]}_{timesteps[test_idx]}.jpg'))
                        
                        save_result(
                            reward_model, processed_out[test_idx][0], bbox_outputs[test_idx], score_thr=score_thr, out_file=os.path.join(score_path, f'fin_out_{batch["idxs"][selected_idx]}_{timesteps[test_idx]}.jpg')
                        )
                return
                
                # output: type -> list(tuple(torch.size[300,5],torch.size[300])) len(output) == batch_size == 32
                # output_meta: type -> list(list(dict)) len(output_meta) == 1 len(output_meta[0]) == batch_size ==32
                
                losses_cls=[]
                losses_bbox=[]
                losses_iou=[]
                losses_bbox_xy=[]
                losses_bbox_hw=[]
                
                bbox_outputs, outputs, output_metas = inference_detector(reward_model, prediction, tensor=True)
                bbox_labels, labels, label_metas = inference_detector(reward_model, ground_truth, tensor=True)
                cls_score = [output[2] for output in outputs]
                bbox_preds = torch.stack([output[0][:,:-1] for output in outputs], dim=0)
                
                img_metas = label_metas[0]
                
                normalized_boxes = []
                for img_meta, pred_bbox in zip(img_metas, bbox_preds):
                    img_h, img_w, _ = img_meta['img_shape']
                    factor = pred_bbox.new_tensor([img_w, img_h, img_w,
                                                img_h]).unsqueeze(0).repeat(
                                                    pred_bbox.size(0), 1)
                    gt_bbox_ = bbox_xyxy_to_cxcywh(pred_bbox)
                    normalized_boxes.append(gt_bbox_ / factor)
                for batch_idx in range(args.train_batch_size):
                    gt_scores = labels[batch_idx][0][:,-1]
                    gt_score_thr = args.score_thr
                    gt_inds = gt_scores > gt_score_thr
                    while(gt_inds.sum()==0):
                        gt_score_thr = gt_score_thr * 0.9
                        gt_inds = gt_scores > gt_score_thr
                    gt_bboxes = labels[batch_idx][0][:,:-1][gt_inds, :]
                    gt_labels = labels[batch_idx][1][gt_inds]
                    
                    pr_score_thr = args.score_thr
                    pr_scores = outputs[batch_idx][0][:,-1]
                    pr_inds = pr_scores > pr_score_thr
                    while(pr_inds.sum()==0):
                        pr_score_thr = pr_score_thr * 0.9
                        pr_inds = pr_scores > pr_score_thr
                    pr_bboxes = normalized_boxes[batch_idx][pr_inds, :]
                    pr_score = cls_score[batch_idx][pr_inds, :]
                    (loss_cls, loss_bbox, loss_iou , loss_bbox_xy, loss_bbox_hw) = rwd_loss_func(
                        cls_scores = pr_score.unsqueeze(0),
                        bbox_preds = pr_bboxes.unsqueeze(0),
                        gt_bboxes_list = [gt_bboxes],
                        gt_labels_list = [gt_labels],
                        img_metas = [img_metas[batch_idx]],
                    )
                    losses_cls.append(loss_cls)
                    losses_bbox.append(loss_bbox)
                    losses_iou.append(loss_iou)
                    losses_bbox_xy.append(loss_bbox_xy)
                    losses_bbox_hw.append(loss_bbox_hw)
                
                losses_cls=torch.stack(losses_cls,dim=0)
                losses_bbox=torch.stack(losses_bbox,dim=0)
                losses_iou=torch.stack(losses_iou,dim=0)
                losses_bbox_xy=torch.stack(losses_bbox_xy,dim=0)
                losses_bbox_hw=torch.stack(losses_bbox_hw,dim=0)                 
                
                reward_loss =  losses_cls + losses_bbox + losses_iou + losses_bbox_xy + losses_bbox_hw
                reward_loss = reward_loss.reshape_as(timestep_mask)
                reward_loss = (timestep_mask * reward_loss).sum() / (timestep_mask.sum() + 1e-10)
                
                # reward_loss = reward_loss.reshape_as(timestep_mask)
                # reward_loss = (timestep_mask * reward_loss).sum() / (timestep_mask.sum() + 1e-10)
                
                # print(reward_loss)
                # import pdb
                # pdb.set_trace()
                # print(f"reward_loss:{reward_loss} - loss_cls:{loss_cls}, loss_bbox:{loss_bbox}, loss_iou:{loss_iou}, loss_bbox_xy:{loss_bbox_xy}, loss_bbox_hw:{loss_bbox_hw}")
                # dot=make_dot(reward_loss, params={"encoder_hidden_states": encoder_hidden_states, "image": image})
                # dot.render("train_graph", format="svg")
                # return
                # reward_loss = (timestep_mask * reward_loss).sum() / (timestep_mask.sum() + 1e-10)
                
                # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                pretrain_loss = (noise_pred.float() - noise.float()) ** 2 # [B, 4, H, W]
                pretrain_loss = pretrain_loss * batch["bbox_mask"] if args.foreground_loss_mode else pretrain_loss
                pretrain_loss = pretrain_loss.mean()
                # print("=========",pretrain_loss,reward_loss,args.grad_scale)
                loss = pretrain_loss + reward_loss * args.grad_scale
                """
                Losses
                """
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                avg_pretrain_loss = accelerator.gather(pretrain_loss.repeat(args.train_batch_size)).mean()
                avg_reward_loss = accelerator.gather(reward_loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_pretrain_loss += avg_pretrain_loss.item() / args.gradient_accumulation_steps
                train_reward_loss += avg_reward_loss.item() / args.gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), filter(lambda p: p.requires_grad, text_encoder.parameters()))
                        if args.train_text_encoder
                        else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                if args.use_ema_text:
                    ema_text_encoder.step(text_encoder.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": train_loss,
                    "train_pretrain_loss": train_pretrain_loss,
                    "train_reward_loss": train_reward_loss, 
                    "lr": lr_scheduler.get_last_lr()[0]
                }, step=global_step)
                train_loss, train_pretrain_loss, train_reward_loss = 0., 0., 0.
                
                loss_per_epoch += train_loss
                pretrain_loss_per_epoch += train_pretrain_loss
                reward_loss_per_epoch += train_reward_loss
                
            logs = {
                "step_loss": loss.detach().item(), 
                "pretrain_loss_step": pretrain_loss.detach().item(),
                "reward_loss_step": reward_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            # save current checkpoint every 500 iterations
            if global_step % args.save_ckpt_freq == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    # copy ema parameters to online model
                    if args.use_ema:
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())
                    if args.use_ema_text:
                        ema_text_encoder.store(text_encoder.parameters())
                        ema_text_encoder.copy_to(text_encoder.parameters())
                    unet_module = accelerator.unwrap_model(unet)
                    pipeline = StableDiffusionPipeline(
                        text_encoder=text_encoder if not args.train_text_encoder else accelerator.unwrap_model(text_encoder),
                        vae=vae,
                        unet=unet_module,
                        tokenizer=tokenizer,
                        scheduler=PNDMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler"),
                        safety_checker=StableDiffusionSafetyChecker.from_pretrained(args.pretrained_model_name_or_path, subfolder="safety_checker"),
                        feature_extractor=CLIPFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor"),
                    )
                    os.makedirs(os.path.join(args.output_dir, args.name, 'iter_' + str(global_step)), exist_ok=True)
                    print("saving into:",os.path.join(args.output_dir, args.name, 'iter_' + str(global_step)))
                    pipeline.save_pretrained(os.path.join(args.output_dir, args.name, 'iter_' + str(global_step)))
                    # restore online parameters
                    if args.use_ema:
                        ema_unet.restore(unet.parameters())
                    if args.use_ema_text:
                        ema_text_encoder.restore(text_encoder.parameters())

    # Create the pipeline using the trained modules and save it.
    # EMA model: not used during training, just record it and save it finally
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        # copy ema parameters to online model
        if args.use_ema:
            ema_unet.store(unet.parameters())
            ema_unet.copy_to(unet.parameters())
        if args.use_ema_text:
            ema_text_encoder.store(text_encoder.parameters())
            ema_text_encoder.copy_to(text_encoder.parameters())        
        # unwrap_model: return model.module of a DDP model
        unet_module = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder if not args.train_text_encoder else accelerator.unwrap_model(text_encoder),
            vae=vae,
            unet=unet_module,
            tokenizer=tokenizer,
            scheduler=PNDMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler"),
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(args.pretrained_model_name_or_path, subfolder="safety_checker"),
            feature_extractor=CLIPFeatureExtractor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor"),
        )
        os.makedirs(os.path.join(args.output_dir, args.name, 'final'), exist_ok=True)
        pipeline.save_pretrained(os.path.join(args.output_dir, args.name, 'final'))
        # restore online parameters
        if args.use_ema:
            ema_unet.restore(unet.parameters())
        if args.use_ema_text:
            ema_text_encoder.restore(text_encoder.parameters())        

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
