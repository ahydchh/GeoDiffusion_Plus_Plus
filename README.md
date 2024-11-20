# GeoDiffusion_Plus_Plus

数据集以及预训练的GeoDiffusion模型 请参考 [GeoDiffusion](https://github.com/KaiChen1998/GeoDiffusion)

[Haohan Chi](https://github.com/ahydchh/ahydchh.github.io) ,[Chenyu Liu](https://github.com/clmdy), [Huan-ang Gao](https://github.com/c7w)

请额外下载COCO官方的instances_val2017.json，也下载到data文件夹下

环境配置方法

### Sparse Head (DETR)

```Bash
conda create -n geo_detr python=3.10 -y
conda activate geo_detr
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd thirdparty/Semi-DETR/thirdparty/mmdetection/			(HEAD detached at v2.27.0)
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

```Bash
conda create -n geo_fcos python=3.10 -y
conda activate geo_fcos
conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
cd thirdparty/Semi-DETR/thirdparty/mmdetection/			(HEAD detached at v2.27.0)
python -m pip install -e .
cd ../../
pip install cython==0.29.33
pip install -r requirements.txt
pip install numpy==1.26.4
```

thirdparty中的[Semi-DETR](https://github.com/JCZ404/Semi-DETR)以及其中的[mmdetection](https://github.com/open-mmlab/mmdetection)是我们做了一定的修改以用于`reward consistency losses`的训练的结果

对于您在训练过程中使用的DETR的pre-trained model，也请通过[Semi-DETR](https://github.com/JCZ404/Semi-DETR)获取；对于FCOS的pre-trained model，请通过[FCOS_weight_download_url](https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth)直接获取，将其重命名为`fcos_r50_caffe_fpn_gn-head_1x_coco.pth`之后放在``GeoDiffusion_Plus_Plus/thirdparty/FCOS/models``文件夹下

还有其他注意事项：因为一些版本问题，您可能需要额外做如下修改

对于您环境下`geo_detr/lib/python3.10/site-packages/transformers/feature_extraction_utils.py`做如下修改

在`line 355`的`get_feature_extractor_dict`中添加`subfolder = kwargs.pop("subfolder", None)`（例如您可以添加在`line 376`），同时对于 `line 402 `的 `cached_file` 函数调用，添加`subfolder=subfolder`

#### train

``Dense:``请运行`tools/dense_reward_dist_train.sh`或`dense_nui_reward_dist_train.sh`

``Sparse:``请运行`tools/sparse_reward_dist_train.sh`或`sparse_nui_reward_dist_train.sh`

模型将保存在`sd-model-finetuned`

#### infer

请运行`tools/infer.sh`

#### fid

请运行修改参数并运行`rebuild.py`生成对应的`data/eval_coco`以及`data/eval_nuimages`文件夹，这将用于测量`fid`

（运行参数可套用`infer.sh`中对应`coco`和`nuimages`的参数）

对于`fid`的测量，请运行`fid.py fid_nui.py`，将读取`dirs.txt`以及`dirs_nui.txt`中生成图片所在的文件夹路径，并将结果保存到`fid.csv fid_nui.csv`

若中间出现问题，可以检查是否是`fid`在下载`inception model`时的问题，如仍有问题，可修改fid代码，手动下载并更改加载路径[inception model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt)

#### mAP

```
conda create -n geo_mAP python=3.10 -y
conda activate geo_mAP
pip install ultralytics
cd thirdparty/yolo
pip install -r requirements.txt  # install
```

我们保留了[yolov5](https://github.com/ultralytics/yolov5)中必要的文件并额外添加了一些文件用于mAP的测量

请运行`eval_map.sh nui_eval_map.sh`

所需要的checkpoint已经包含在仓库中，当然您也可以根据[yolov5](https://github.com/ultralytics/yolov5)中的指引使用其他模型或者fine-tune任何您需要的模型
