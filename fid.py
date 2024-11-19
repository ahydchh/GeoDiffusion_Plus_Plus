# pip3 install clean-fid
# https://github.com/GaParmar/clean-fid
from cleanfid import fid
import argparse
import csv
from tqdm import tqdm
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def eval(real_image_path, generated_image_path):
    score = 0.0
    # We have 5 groups of generated images
    for i in range(5):
        score += fid.compute_fid(
            real_image_path,
            f'{generated_image_path}/group_{i}',
            dataset_res=512,
            batch_size=128
        )
    # Report the average FID score
    return (score / 5)
   
with open('dirs.txt','r') as ds:
    dirs = ds.readlines()
    
results = {}
csv_file = 'fid.csv'
for d in tqdm(dirs):
    dir = d.strip()
    score = eval('data/eval_coco',f'work_dirs/output_img/{dir}/val/val-100_scale5_samples5')
    results[dir] = score
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 写入一个键值对
        writer.writerow([dir, score])
print(results)
