# pip3 install clean-fid
# https://github.com/GaParmar/clean-fid
from cleanfid import fid
import argparse
import csv
from tqdm import tqdm
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def eval(real_image_path, generated_image_path, nsamples):
    score = 0.0
    for grp in range(nsamples):
        grp_dir = os.path.join(generated_image_path, f"group_{grp}")
        grp_s = []
        for cls in ["CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT",'CAM_FRONT','CAM_FRONT_LEFT',"CAM_FRONT_RIGHT"]:
            print(os.path.join(real_image_path, f"samples/{cls}"))
            print(os.path.join(grp_dir, f"samples/{cls}"))
            grp_s.append(fid.compute_fid(
                os.path.join(real_image_path, f"samples/{cls}"),
                os.path.join(grp_dir, f"samples/{cls}"),
                dataset_res=256,
                batch_size=128
            ))
        score += sum(grp_s) / len(grp_s)
    # Report the average FID score
    return (score / nsamples)
   
with open('dirs_nui.txt','r') as ds:
    dirs = ds.readlines()
    
results = {}
csv_file = 'fid_nui.csv'
for d in tqdm(dirs):
    dir = d.strip()
    if '/' not in dir:
        target_dir = f'work_dirs/output_img/{dir}/val/'
        target_dir = os.path.join(target_dir, os.listdir(target_dir)[0])
        num_samples = int(target_dir[-1])
        score = eval('data/eval_nuimages',target_dir, num_samples)
    else:
        f'{dir}/val/'
        target_dir = os.path.join(target_dir, os.listdir(target_dir)[0])
        num_samples = int(target_dir[-1])
        score = eval('data/eval_nuimages',target_dir, num_samples)
    results[dir] = score
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 写入一个键值对
        writer.writerow([dir, score])
print(results)
