CKPT_PATH=sd-model-finetuned/test/iter_
accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --gpu_ids 4,5,6,7 --num_processes=4 test_geodiffusion.py ${CKPT_PATH}10 --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 --nsamples 5 ${PY_ARGS}
# accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --gpu_ids 4,5,6,7 --num_processes=4 test_geodiffusion.py ${CKPT_PATH}10 --dataset_config_name configs/data/nuimage_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 --nsamples 5 ${PY_ARGS}
