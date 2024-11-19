PY_ARGS=${@:1}
EPOCHS=3
NAME="finetune_lr_5e-6_nowarm"
SAVE_FREQ=1000
MODEL=/data2/chihh/chihh/GeoDiffusion/geodiffusion-coco-stuff-256x256
accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --mixed_precision fp16 --gpu_ids 6,7 --num_processes 2 \
train_geodiffusion.py \
    --pretrained_model_name_or_path ${MODEL} \
    --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder \
    --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
    --seed 0 --train_batch_size 16 --gradient_accumulation_steps 1 --gradient_checkpointing \
    --mixed_precision fp16 --num_train_epochs 60 --learning_rate 5e-6 --max_grad_norm 1 \
    --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine --lr_warmup_steps 0 \
    --dataset_config_name configs/data/coco_stuff_256x256.py \
    --uncond_prob 0.1 \
    --name ${NAME} \
    --save_ckpt_freq ${SAVE_FREQ} \
    --num_train_epochs ${EPOCHS} \
    ${PY_ARGS}