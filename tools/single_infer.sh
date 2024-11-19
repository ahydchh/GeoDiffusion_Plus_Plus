# CKPT_PATH=/gpfs/essfs/iat/Tsinghua/chihh/GeoDiffusion/sd-model-finetuned/base/iter_100000
export CUDA_VISIBLE_DEVICES=1
CKPT_PATH=geodiffusion-coco-stuff-256x256
python single_test.py $CKPT_PATH --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}