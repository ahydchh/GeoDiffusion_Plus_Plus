# GeoDiffusion_Plus_Plus

The dataset and pretrained GeoDiffusion model can be found at [GeoDiffusion](https://github.com/KaiChen1998/GeoDiffusion).

Additionally, download the COCO official `instances_val2017.json` and place it in the `data` folder.

## Environment Setup

### Sparse Head (DETR)

```bash
conda create -n geo_detr python=3.10 -y
conda activate geo_detr
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd thirdparty/Semi-DETR/thirdparty/mmdetection/   # (HEAD detached at v2.27.0)
python -m pip install -e .
cd ../../
python -m pip install -e .
cd detr_od/models/utils/ops
python setup.py build install
cd ../../../../../../
pip install cython==0.29.33
pip install -r requirements.txt
pip install numpy==1.26.4
```

### Dense Head (FCOS)

```bash
conda create -n geo_fcos python=3.10 -y
conda activate geo_fcos
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd thirdparty/FCOS   # (modified from mmdetection HEAD detached at v2.27.0)
python -m pip install -e .
cd ../../
pip install cython==0.29.33
pip install -r requirements.txt
pip install numpy==1.26.4
```

The [Semi-DETR](https://github.com/JCZ404/Semi-DETR) and [mmdetection](https://github.com/open-mmlab/mmdetection) repositories in `thirdparty` have been modified for reward consistency losses during training.

For the pretrained model of DETR, refer to [Semi-DETR](https://github.com/JCZ404/Semi-DETR). For FCOS pretrained models, download directly from [FCOS_weight_download_url](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth). Rename it to `fcos_r50_caffe_fpn_gn-head_1x_coco.pth` and place it in the `GeoDiffusion_Plus_Plus/thirdparty/FCOS/models` folder.

### Additional Notes
Due to version-related issues, you might need to make the following modifications:

- In `geo_detr/lib/python3.10/site-packages/transformers/feature_extraction_utils.py`:
  - Add `subfolder = kwargs.pop("subfolder", None)` in the `get_feature_extractor_dict` function (e.g., after line 355).
  - Modify the `cached_file` function call on line 402 to include `subfolder=subfolder`.

---

## Training

### Dense
Run:  
```bash
tools/dense_reward_dist_train.sh
```  
or:  
```bash
tools/dense_nui_reward_dist_train.sh
```

### Sparse
Run:  
```bash
tools/sparse_reward_dist_train.sh
```  
or:  
```bash
tools/sparse_nui_reward_dist_train.sh
```

The models will be saved in `sd-model-finetuned`.

---

## Inference

Run:
```bash
tools/infer.sh
```

---

## FID Measurement

1. Modify parameters and run `rebuild.py` to generate the corresponding `data/eval_coco` and `data/eval_nuimages` folders. These will be used for FID measurement.  
   (You can reuse the parameters from `infer.sh` for COCO and NuImages.)

2. To measure FID, run:
   ```bash
   fid.py
   fid_nui.py
   ```

   These scripts will read the folder paths specified in `dirs.txt` and `dirs_nui.txt` and save the results in `fid.csv` and `fid_nui.csv`.

If any issues arise (e.g., downloading the Inception model for FID), manually download the [Inception model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt) and update the loading path in the clean-fid packet.

---

## mAP Measurement

1. Environment setup:
   ```bash
   conda create -n geo_mAP python=3.10 -y
   conda activate geo_mAP
   pip install ultralytics
   cd thirdparty/yolo
   pip install -r requirements.txt  # install
   ```

2. We retained necessary files from [yolov5](https://github.com/ultralytics/yolov5) and added additional files for mAP measurement.

3. To measure mAP, run:
   ```bash
   eval_map.sh
   nui_eval_map.sh
   ```

   The required checkpoint is included in the repository. Alternatively, you can follow [yolov5](https://github.com/ultralytics/yolov5) instructions to use other models or fine-tune any model of your choice.