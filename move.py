import os
import shutil
from tqdm import tqdm
# 指定你想要处理的文件夹路径
with open('dirs.txt','r') as ds:
    dirs = ds.readlines()
for dir_name in dirs:
    dir_name = dir_name.strip()
    source_folder = f'work_dirs/output_img/{dir_name}/val/val-100_scale5_samples5'
    for index in range(5):
        target_folder = os.path.join(source_folder, f'group_{index}')
            
        # 如果目标文件夹不存在，则创建
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            
    # 遍历指定文件夹中的所有文件
    for filename in tqdm(os.listdir(source_folder)):
        # 检查文件名是否符合命名规则
        if filename.endswith(".jpg") and "_" in filename:
            # 解析文件名，获取前面的部分和i值
            name_part, index = filename.rsplit('_', 1)
            index = index.split('.')[0]  # 去掉文件扩展名的部分
            
            # 创建目标文件夹路径
            target_folder = os.path.join(source_folder, f'group_{index}')
            
            # 构建目标文件路径，新文件名去掉i部分
            new_filename = f'{name_part}.jpg'
            target_path = os.path.join(target_folder, new_filename)
            
            # 移动并重命名文件
            source_path = os.path.join(source_folder, filename)
            shutil.move(source_path, target_path)
