export CUDA_VISIBLE_DEVICES=7
CKPT_PATH=geodiffusion-nuimages-256x256
python single_test.py $CKPT_PATH --dataset_config_name configs/data/nuimage_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}
