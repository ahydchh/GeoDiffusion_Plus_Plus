CUDA_VISIBLE_DEVICES=1 python eval_nui_map.py --weights finetuned_nui.pt --anno ../../data/nuimages/annotation/val/nuimages_v1.0-val.json --image_dir ../../data/eval_nuimages --save_csv_path ../../nui_mAP.csv --dir dirs_nui.txt