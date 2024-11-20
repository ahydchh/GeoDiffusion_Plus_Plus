# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import asyncio
import glob
import os
from argparse import ArgumentParser
import numpy as np
from PIL import Image
from mmcv import Config
from mmdet.apis import async_inference_detector, inference_detector, show_result_pyplot
import mmcv
from detr_ssod.apis.inference import init_detector, save_result
from detr_ssod.utils import patch_config
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from mmcv.parallel import DataContainer as DC

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--img", help="Image file")
    parser.add_argument("--config", help="Config file")
    parser.add_argument("--checkpoint", help="Checkpoint file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--score_thr", type=float, default=0.3, help="bbox score threshold"
    )
    parser.add_argument(
        "--async-test",
        action="store_true",
        help="whether to set async options for async inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="specify the directory to save visualization results.",
    )
    args = parser.parse_args()
    return args

def test_pipeline(img):
    if not isinstance(img, torch.Tensor):
        # Load
        client = mmcv.FileClient(backend='disk')
        img_bytes = client.get(img)
        numpy_img = mmcv.imfrombytes(
                img_bytes, flag='color', channel_order='bgr')
        ori_img = torch.tensor(numpy_img,requires_grad=True,dtype=torch.float32)
    # print(numpy_img)
    # image = Image.fromarray(numpy_img, mode='RGB')
    # image.save(os.path.join(f'test_dir/out_start.jpg'))

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
    
    # image = img.permute(1,2,0).detach().cpu().numpy()
    # image = Image.fromarray(image.astype(numpy_img.dtype), mode='RGB')
    # image = image.resize((img_width, img_height))
    # image.save(os.path.join(f'test_dir/out_resize.jpg'))
    
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
    # mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float64, device=results['img'].device).reshape(1, -1)
    # stdinv = 1 / torch.tensor([58.395, 57.12, 57.375], dtype=results['img'].dtype, device=results['img'].device).reshape(1, -1)
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
    # image = results['img'].permute(1,2,0).detach().cpu().numpy()
    # image = Image.fromarray(image.astype(numpy_img.dtype), mode='RGB')
    # image = image.resize((img_width, img_height))
    # image.save(os.path.join(f'test_dir/out_norm.jpg'))

    # mean = torch.tensor([123.675, 116.28, 103.53])  # 根据实际情况设置
    # std = torch.tensor([58.395, 57.12, 57.375])   # 根据实际情况设置
    # # 要确保 mean 和 std 维度正确
    # mean = mean.view(3, 1, 1)  # 如果图像是 [C, H, W] 的张量
    # std = std.view(3, 1, 1)
    # image = results['img'].detach() * std + mean
    # image = image.permute(1,2,0).cpu().numpy()
    # image = Image.fromarray(image.astype(numpy_img.dtype), mode='RGB')
    # # image = image.resize((img_width, img_height))
    # image.save(os.path.join(f'test_dir/out_renorm.jpg'))
    
    results['img'] = DC(results['img'], padding_value=0, stack=True)
    # Collect 
    data={}
    img_meta = {}
    for key in ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg']:
        img_meta[key] = results[key]
    data['img_metas']=DC(img_meta,cpu_only=True)
    data['img']=results['img']
    data = {'img':[data['img']], 'img_metas':[data['img_metas']]}
    print(numpy_img.dtype)
    return numpy_img, data, ori_img

def main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    
    result = test_pipeline(args.img)
    numpy_img = result[0]
    imgs = result[1]

    bbox_result, result, meta = inference_detector(model, imgs, tensor=True)

    if args.output is None:
        show_result_pyplot(model, args.img, bbox_result, score_thr=args.score_thr)
    else:
        out_file_path = os.path.join(args.output, os.path.basename(args.img))
        print(f"Save results to {out_file_path}")
        save_result(
            model, numpy_img, bbox_result[0], score_thr=args.score_thr, out_file=out_file_path
        )


async def async_main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    # test a single image
    args.img = glob.glob(args.img)
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    for img, pred in zip(args.img, result):
        if args.output is None:
            show_result_pyplot(model, img, pred, score_thr=args.score_thr)
        else:
            out_file_path = os.path.join(args.output, os.path.basename(img))
            print(f"Save results to {out_file_path}")
            save_result(
                model, img, pred, score_thr=args.score_thr, out_file=out_file_path
            )


if __name__ == "__main__":
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)