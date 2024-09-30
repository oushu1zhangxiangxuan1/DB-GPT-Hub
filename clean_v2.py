import os
import shutil

base_dir = "/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter"

# 遍历base_dir下面一层的文件夹
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

print(subdirs)

for subdir in subdirs:
    subdir_path = os.path.join(base_dir, subdir)
    
    # 获取子文件夹下中的checkpoint-{num}开头的所有文件夹
    checkpoint_dirs = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d)) and d.startswith("checkpoint-")]
    
    if not checkpoint_dirs:
        continue
    
    # 找到num最大和倒数第二大的文件夹
    sorted_checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
    max_checkpoint_dir = sorted_checkpoint_dirs[-1]
    second_max_checkpoint_dir = sorted_checkpoint_dirs[-2]

    # 删除 global_step_{num}的文件夹
    for d in checkpoint_dirs:
        if d != max_checkpoint_dir and d != second_max_checkpoint_dir:
            global_step_dir = os.path.join(subdir_path, d, f"global_step{d.split('-')[1]}")
            if os.path.exists(global_step_dir):
                shutil.rmtree(global_step_dir)
                print(f"删除文件夹：{global_step_dir}")

    # 输出被删除文件夹的绝对路径
    max_checkpoint_dir_path = os.path.join(subdir_path, max_checkpoint_dir)
    print(f"最大的文件夹：{max_checkpoint_dir_path}")
