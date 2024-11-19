# PY_ARGS=${@:1}
EPOCHS=3
NAME="ablation_score0.3_reward0.01_lr1e-5_sample1000_reward200_finetune"
SAVE_FREQ=1000
MODEL_PATH=geodiffusion-coco-stuff-256x256
# MODEL_PATH=stable-diffusion-v1-5/stable-diffusion-v1-5
# accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --mixed_precision fp16 --gpu_ids 1,2,3 --num_processes 4 \
# reward_train_geodiffusion.py \
#     --pretrained_model_name_or_path ${MODEL_PATH} \
#     --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder \
#     --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
#     --seed 0 --train_batch_size 8 --gradient_accumulation_steps 1 --gradient_checkpointing \
#     --mixed_precision fp16 --learning_rate 1e-5 --max_grad_norm 1 \
#     --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine \
#     --dataset_config_name configs/data/coco_stuff_256x256.py \
#     --uncond_prob 0.1 \
#     --name ${NAME} \
#     --save_ckpt_freq ${SAVE_FREQ} \
#     --num_train_epochs ${EPOCHS} \
#     --reward_config ../Semi-DETR/configs/detr_ssod/detr_ssod_dino_detr_r50_coco_full_240k.py \
#     --grad_scale 0.01 \
#     --diff_scale 0.0 \
#     --min_timestep_rewarding 0 \
#     --max_timestep_rewarding 250 \
#     --timestep_sampling_start 0 \
#     --timestep_sampling_end 1000 \
#     --lr_warmup_steps 1000 \
#     --score_thr 0.3
#     # ${PY_ARGS}

CKPT_PATH=sd-model-finetuned/${NAME}/iter_

accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --gpu_ids 1,2,3 --num_processes=3 test_geodiffusion.py ${CKPT_PATH}10000 --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 --nsamples 1 ${PY_ARGS}

accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --gpu_ids 1,2,3 --num_processes=3 test_geodiffusion.py ${CKPT_PATH}5000 --dataset_config_name configs/data/coco_stuff_256x256.py --prompt_version v1 --num_bucket_per_side 256 256 --nsamples 1 ${PY_ARGS}
