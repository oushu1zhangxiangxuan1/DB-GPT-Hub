#!/bin/bash
source ~/.bash_profile

conda_envs=(
    "/mnt/datadisk0/conda_envs/baai"
    "/mnt/datadisk0/conda_envs/glm"
    "/root/space/conda_envs/dbgpt"
    "/root/space/conda_envs/dbgpt-coding"
    "/root/space/conda_envs/dbgpt_hub"
    "/root/space/conda_envs/dt42"
    "/root/space/conda_envs/fc232"
    "/root/space/conda_envs/fctest"
    "/root/space/conda_envs/hubc12"
    "/root/space/conda_envs/slora"
    "/root/space/conda_envs/t2c12"
)
# set -x
for env in "${conda_envs[@]}"
do
    echo "Checking $env..."
    conda activate $env
    if pip list | grep -q "vllm"; then
        vllm_version=$(pip show vllm | grep Version | awk '{print $2}')
        echo "Found vllm in $env, version: $vllm_version"
    else
        echo "vllm not found in $env"
    fi
    conda deactivate
    # break
    echo '\n\n'
done


    # # 保存 pip list 输出到变量
    # pip_list_output=$(pip list)

    # if echo "$pip_list_output" | grep -q "vllm"; then
    #     vllm_version=$(echo "$pip_list_output" | grep vllm | awk '{print $2}')
