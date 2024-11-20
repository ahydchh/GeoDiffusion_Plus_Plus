# GeoDiffusion_Plus_Plus

数据集以及预训练的GeoDiffusion模型 请参考 [GeoDiffusion](https://github.com/KaiChen1998/GeoDiffusion)

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

还有其他注意事项：因为一些版本问题，你需要额外做如下修改

对于您环境下geo_detr/lib/python3.10/site-packages/transformers/feature_extraction_utils.py做如下修改

在line 355的get_feature_extractor_dict中添加subfolder = kwargs.pop("subfolder", None)（例如您可以添加在line 376），同时对于 line 402 行的 cached_file 函数调用，添加subfolder=subfolder

#### train

请运行tools/reward_dist_train.sh或nui_reward_dist_train.sh

模型将保存在sd-model-finetuned

#### infer

请运行tools/infer.sh

#### eval

##### fid

请运行修改参数并运行rebuild.py生成对应的data/eval_coco以及data/eval_nuimages文件夹，这将用于测量fid

（运行参数可套用infer.sh中对应coco和nuimages的参数）

对于fid的测量，请运行fid.py fid_nui.py，将读取dirs.txt以及dirs_nui.txt中生成图片所在的文件夹路径，并将结果保存到fid.csv fid_nui.csv

若中间出现问题，可以检查是否是fid在下载inception model时的问题，如仍有问题，可修改fid代码，手动下载并更改加载路径[inception model](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt)

##### mAP

```
conda create -n geo_mAP python=3.10 -y
conda activate geo_mAP
pip install ultralytics
cd thirdparty/yolo
pip install -r requirements.txt  # install
```

请运行eval_map.sh nui_eval_map.sh

