# Copyright (c) OpenMMLab. All rights reserved.
# Modified from thirdparty/mmdetection/demo/image_demo.py
import glob
import os
from argparse import ArgumentParser
import torch
from mmcv import Config
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.core import bbox_xyxy_to_cxcywh
from detr_ssod.apis.inference import init_detector, save_result
from detr_ssod.utils import patch_config
import glob
import os
from argparse import ArgumentParser
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


def main(args):
    cfg = Config.fromfile(args.config)
    # Not affect anything, just avoid index error
    cfg.work_dir = "./work_dirs"
    cfg = patch_config(cfg)
    # build the model from a config file and a checkpoint file
    model = init_detector(cfg, args.checkpoint, device=args.device)
    
    img = "resized_example.jpg"
    img1 = "resized_example1.jpg"
    
    numpy_img, data, ori_img = test_pipeline(img)
    numpy_img1, data1, ori_img1 = test_pipeline(img1)
    
    bbox_label, label, label_meta = inference_detector(model, data, tensor=True)
    bbox_output , output, output_meta = inference_detector(model, data1, tensor=True)

    cls_score = output[0][2]
    bbox_head = model.model().bbox_head
    bbox_preds = output[0][0][:,:-1].unsqueeze(0)
    img_metas = [label_meta[0][0]]
    normalized_boxes = []
    for img_meta, pred_bbox in zip(img_metas, bbox_preds):
        img_h, img_w, _ = img_meta['img_shape']
        factor = pred_bbox.new_tensor([img_w, img_h, img_w,
                                    img_h]).unsqueeze(0).repeat(
                                        pred_bbox.size(0), 1)
        gt_bbox_ = bbox_xyxy_to_cxcywh(pred_bbox)
        normalized_boxes.append(gt_bbox_ / factor)
    
    normalized_boxes = torch.stack(normalized_boxes, dim=0)

    (loss_cls, loss_bbox, loss_iou , loss_bbox_xy, loss_bbox_hw) = bbox_head.loss_single(
        cls_scores = cls_score.unsqueeze(0),
        bbox_preds = normalized_boxes,
        gt_bboxes_list = [label[0][0][:,:-1]],
        gt_labels_list = [label[0][1]],
        img_metas = img_metas,
    )
    # print(type(label),type(output))
    # print(label.shape,output.shape)
    print(loss_cls, loss_bbox, loss_iou, loss_bbox_xy, loss_bbox_hw)
    save_result(
                model, "resized_example.jpg", bbox_label[0], score_thr=args.score_thr, out_file="output/resized_example.jpg"
            )
    save_result(
                model, "resized_example1.jpg", bbox_output[0], score_thr=args.score_thr, out_file="output/resized_example1.jpg"
            )
    
if __name__ == "__main__":
    args = parse_args()
    main(args)