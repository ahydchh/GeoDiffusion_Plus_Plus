CKPT_PATH=sd-model-finetuned/score0.3_reward0.01_lr1e-5_sample1000_reward250_finetune/iter_10000
accelerate launch --multi_gpu --main_process_port=$(python random_port.py)  --gpu_ids 6,7 --num_processes=2 test_geodiffusion.py $CKPT_PATH --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}

CKPT_PATH=sd-model-finetuned/score0.3_reward0.01_lr1e-5_sample1000_reward250_finetune/iter_6000
accelerate launch --multi_gpu --main_process_port=$(python random_port.py)  --gpu_ids 6,7 --num_processes=2 test_geodiffusion.py $CKPT_PATH --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}

CKPT_PATH=sd-model-finetuned/score0.3_reward0.01_lr1e-5_sample1000_reward250_finetune/iter_3000
accelerate launch --multi_gpu --main_process_port=$(python random_port.py)  --gpu_ids 6,7 --num_processes=2 test_geodiffusion.py $CKPT_PATH --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}

CKPT_PATH=sd-model-finetuned/score0.3_reward0.01_lr1e-5_sample1000_reward250_finetune/iter_1000
accelerate launch --multi_gpu --main_process_port=$(python random_port.py)  --gpu_ids 6,7 --num_processes=2 test_geodiffusion.py $CKPT_PATH --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}

CKPT_PATH=sd-model-finetuned/score0.3_reward0.01_lr1e-5_sample1000_reward250_finetune/iter_8000
accelerate launch --multi_gpu --main_process_port=$(python random_port.py)  --gpu_ids 6,7 --num_processes=2 test_geodiffusion.py $CKPT_PATH --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 ${PY_ARGS}