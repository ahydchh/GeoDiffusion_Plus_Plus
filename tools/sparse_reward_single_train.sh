PORT=${PORT:-29501}
EPOCHS=3
NAME="test_"
SAVE_FREQ=1000
BATCH_SIZE=8
# MODEL_PATH=geodiffusion-coco-stuff-256x256
MODEL_PATH=geodiffusion-coco-stuff-256x256
CUDA_VISIBLE_DEVICES=7 python sparse_train_geodiffusion.py --reward_config thirdparty/Semi-DETR/configs/detr_ssod/detr_ssod_dino_detr_r50_coco_full_240k.py --name ${NAME} --save_ckpt_freq ${SAVE_FREQ} --num_train_epochs ${EPOCHS} --pretrained_model_name_or_path geodiffusion-coco-stuff-256x256 --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm --seed 0 --train_batch_size ${BATCH_SIZE} --gradient_accumulation_steps 1 --gradient_checkpointing --mixed_precision fp16 --learning_rate 1.5e-4 --max_grad_norm 1 --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine --lr_warmup_steps 3000 --dataset_config_name configs/data/coco_stuff_256x256.py --uncond_prob 0.1