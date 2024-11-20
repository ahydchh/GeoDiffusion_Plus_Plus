# pip3 install clean-fid
# https://github.com/GaParmar/clean-fid
from cleanfid import fid
import argparse
import csv
from tqdm import tqdm
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def eval(real_image_path, generated_image_path, num_samples):
    score = 0.0
    # We have 5 groups of generated images
    for i in range(num_samples):
        score += fid.compute_fid(
            real_image_path,
            f'{generated_image_path}/group_{i}',
            dataset_res=256,
            batch_size=128
        )
    # Report the average FID score
    return (score / num_samples)
   
with open('dirs.txt','r') as ds:
    dirs = ds.readlines()
    
results = {}
csv_file = 'fid.csv'
for d in tqdm(dirs):
    dir = d.strip()
    target_dir = f'{dir}/val/'
    target_dir = os.path.join(target_dir, os.listdir(target_dir)[0])
    num_samples = int(target_dir[-1])
    score = eval('data/eval_coco',target_dir, num_samples)
    results[dir] = score
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 写入一个键值对
        writer.writerow([dir, score])
print(results)
