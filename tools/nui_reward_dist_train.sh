# PY_ARGS=${@:1}
EPOCHS=3
NAME="test_nui"
SAVE_FREQ=10
MODEL_PATH=geodiffusion-nuimages-256x256
# MODEL_PATH=stable-diffusion-v1-5/stable-diffusion-v1-5
accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --mixed_precision fp16 --gpu_ids 4,5,6,7 --num_processes 4 \
reward_train_geodiffusion.py \
    --pretrained_model_name_or_path ${MODEL_PATH} \
    --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder \
    --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
    --seed 0 --train_batch_size 8 --gradient_accumulation_steps 1 --gradient_checkpointing \
    --mixed_precision fp16 --learning_rate 1e-7 --max_grad_norm 1 \
    --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine \
    --dataset_config_name configs/data/nuimage_256x256.py \
    --uncond_prob 0.1 \
    --name ${NAME} \
    --save_ckpt_freq ${SAVE_FREQ} \
    --num_train_epochs ${EPOCHS} \
    --reward_config thirdparty/Semi-DETR/configs/dino_detr/dino_detr_r50_8x2_12e_nuimage.py \
    --reward_checkpoint thirdparty/Semi-DETR/models/detr_nui.pth \
    --grad_scale 0.0001 \
    --min_timestep_rewarding 0 \
    --max_timestep_rewarding 300 \
    --timestep_sampling_start 0 \
    --timestep_sampling_end 1000 \
    --lr_warmup_steps 1000 \
    --score_thr 0.7
    # ${PY_ARGS}
