# PY_ARGS=${@:1}
EPOCHS=3
NAME="nui_dense_test"  # 这里也要改！
SAVE_FREQ=1000
MODEL_PATH=geodiffusion-coco-stuff-256x256
# 看一下gpu_id是什么
accelerate launch --multi_gpu --main_process_port=$(python random_port.py) --mixed_precision fp16 --gpu_ids 0,1 --num_processes 2 \
dense_train_geodiffusion.py \
    --pretrained_model_name_or_path ${MODEL_PATH} \
    --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed --train_text_encoder \
    --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
    --seed 0 --train_batch_size 16 --gradient_accumulation_steps 1 --gradient_checkpointing \
    --mixed_precision fp16 --learning_rate 5e-7 --max_grad_norm 1 \
    --lr_text_layer_decay 0.95 --lr_text_ratio 0.75 --lr_scheduler cosine --lr_warmup_steps 500 \
    --dataset_config_name configs/data/coco_stuff_256x256.py \
    --uncond_prob 0.1 \
    --name ${NAME} \
    --centerness_weight 1.0 \
    --reward_scale 0.001\
    --save_ckpt_freq ${SAVE_FREQ} \
    --num_train_epochs ${EPOCHS} \
    --min_timestep_rewarding 0 \
    --max_timestep_rewarding 200 \
    --score_thr 0.40 \
    --reward_config thirdparty/FCOS/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py \
    --reward_checkpoint thirdparty/FCOS/models/fcos_r50_caffe_fpn_gn-head_1x_coco.pth
    # ${PY_ARGS}