#!/bin/bash


checkpoint_count_list=(600 1250 4700)  # 请根据实际需求修改这个列表
model="CodeLlama-13b-Instruct-hf-ds-20240202_194746-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-qb_4qt_fp4-DBTHub_Prompt-chase-WB-FIO-SF"
# 循环执行Python命令
for checkpoint_count in "${checkpoint_count_list[@]}"
do
    python dbgpt_hub/train/export_model.py \
        --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
        --finetuning_type lora \
        --template llama2 \
        --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/$model/checkpoint-$checkpoint_count \
        --output_dir /root/space/models/FineTuned/$model-checkpoint-$checkpoint_count-nofp16/
done


# checkpoint_count_list=(1000 1200 1400)  # 请根据实际需求修改这个列表

# # 循环执行Python命令
# for checkpoint_count in "${checkpoint_count_list[@]}"
# do
#     python dbgpt_hub/train/export_model.py \
#         --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
#         --finetuning_type lora \
#         --template llama2 \
#         --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-ds-20240129_193201-tpl_llama2-src_12000-tgt_5000-r_32-a_8-lt_q_proj-DBTHub_Prompt-chase-WB-FIO/checkpoint-$checkpoint_count \
#         --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-ds-20240129_193201-tpl_llama2-src_12000-tgt_5000-r_32-a_8-lt_q_proj-DBTHub_Prompt-chase-WB-FIO-checkpoint-$checkpoint_count-nofp16/
# done


# checkpoint_count_list=(1000 1200 1250 1500 1800)  # 请根据实际需求修改这个列表

# # 循环执行Python命令
# for checkpoint_count in "${checkpoint_count_list[@]}"
# do
#     python dbgpt_hub/train/export_model.py \
#         --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
#         --finetuning_type lora \
#         --template llama2 \
#         --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-20240129_185740-tpl_llama2_code_hub_1-src_4096-tgt_512-r_32-a_8-lt_q_proj-NULL_Prompt_Inst-chase-WB-FIO/checkpoint-$checkpoint_count \
#         --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-20240129_185740-tpl_llama2_code_hub_1-src_4096-tgt_512-r_32-a_8-lt_q_proj-NULL_Prompt_Inst-chase-WB-FIO-checkpoint-$checkpoint_count-nofp16/
# done


# checkpoint_count_list=(1000 1200 1250 1500 1800)  # 请根据实际需求修改这个列表

# # 循环执行Python命令
# for checkpoint_count in "${checkpoint_count_list[@]}"
# do
#     python dbgpt_hub/train/export_model.py \
#         --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
#         --finetuning_type lora \
#         --template llama2 \
#         --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-20240129_183722-tpl_llama2-src_4096-tgt_512-r_32-a_8-NULL_Prompt_Inst-chase-WB-FIO/checkpoint-$checkpoint_count \
#         --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-20240129_183722-tpl_llama2-src_4096-tgt_512-r_32-a_8-NULL_Prompt_Inst-chase-WB-FIO-checkpoint-$checkpoint_count-nofp16/
# done


# 预定义的checkpoint数列表
# checkpoint_count_list=(1200 10650 10600)  # 请根据实际需求修改这个列表

# # 循环执行Python命令
# for checkpoint_count in "${checkpoint_count_list[@]}"
# do
#     python dbgpt_hub/train/export_model.py \
#         --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
#         --finetuning_type lora \
#         --template llama2 \
#         --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-src_4096-tgt_512-r_32-a_8-NULL_Prompt_Inst-chase-WB-FIO/checkpoint-$checkpoint_count \
#         --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-src_4096-tgt_512-r_32-a_8-NULL_Prompt_Inst-chase-WB-FIO-checkpoint-$checkpoint_count-nofp16/
# done


# checkpoint_count_list=(1250 1200 9600 9550)  # 请根据实际需求修改这个列表
# for checkpoint_count in "${checkpoint_count_list[@]}"
# do
#     python dbgpt_hub/train/export_model.py \
#         --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
#         --finetuning_type lora \
#         --template llama2 \
#         --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-src_4096-tgt_512-r_32-a_8-NULL_Prompt_Inst-chase-WB-FIO-UseSQL/checkpoint-$checkpoint_count \
#         --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-src_4096-tgt_512-r_32-a_8-NULL_Prompt_Inst-chase-WB-FIO-UseSQL-checkpoint-$checkpoint_count-nofp16/
# done
    

# checkpoint_count_list=(1250 1200 10100 10150)  # 请根据实际需求修改这个列表
# for checkpoint_count in "${checkpoint_count_list[@]}"
# do
#     python dbgpt_hub/train/export_model.py \
#         --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
#         --finetuning_type lora \
#         --template llama2 \
#         --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-src_4096-tgt_512-r_32-a_8-DBTHub_Prompt-chase-WB-FIO/checkpoint-$checkpoint_count \
#         --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-src_4096-tgt_512-r_32-a_8-DBTHub_Prompt-chase-WB-FIO-checkpoint-$checkpoint_count-nofp16/
# done