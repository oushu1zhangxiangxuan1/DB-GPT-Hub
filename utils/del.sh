#!/bin/bash

# 定义路径
base_path="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/Skywork-13B-DBTHub_Prompt-spider_chase_chase_cosql_sparc-WB-FIO/"

# 获取所有子文件夹
sub_folders=$(find "$base_path" -maxdepth 1 -mindepth 1 -type d)

# 遍历子文件夹
for folder in $sub_folders; do
    # 获取文件夹名
    folder_name=$(basename "$folder")
    # 尝试将文件夹名解析为数字
    num=$(echo "$folder_name" | awk -F'-' '{print $NF}')
    if [[ $num =~ ^[0-9]+$ ]]; then
        # 如果文件夹名能够解析为数字
        if (( $num % 50 != 0 )); then
            echo "Deleting folder: $folder"
            rm -rf "$folder"
        fi
    fi
done

echo "Done."
