# llama2 series
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /home/LLM/CodeLlama-13b-Instruct-hf \
    --template llama2 \
    --finetuning_type lora \
    --checkpoint_dir dbgpt_hub/output/adapter/CodeLlama-13b-sql-lora \
    --output_dir dbgpt_hub/output/codellama-13b-sql-sft \
    --fp16


## Baichuan2
# python dbgpt_hub/train/export_model.py \
#     --model_name_or_path /root/space/models/baichuan-inc/Baichuan2-13B-Chat/main \
#     --template baichuan2 \
#     --finetuning_type lora \
#     --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/baichuan2-13b-qlora-MetadataPrompt_1_1-UseSQL/checkpoint-1200 \
#     --output_dir /root/space/models/FineTuned/baichuan2-13b-qlora-MetadataPrompt_1_1-UseSQL-checkpoint-1200-$(date "+%Y%m%d%H%M%S")/ \
#     --fp16
#     # --bf16

## AquilaCode-multi
# python dbgpt_hub/train/export_model.py \
#     --model_name_or_path /root/space/models/BAAI/AquilaCode-multi/main \
#     --template aquila \
#     --finetuning_type lora \
#     --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/AquilaCode-DBTHub_Prompt-spider_chase_chase_cosql_sparc-WB-FIO/checkpoint-1250 \
#     --output_dir /root/space/models/FineTuned/AquilaCode-DBTHub_Prompt-spider_chase_chase_cosql_sparc-WB-FIO-checkpoint-1250-$(date "+%Y%m%d%H%M%S")/ \
#     --fp16


## CodeLLaMA
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
    --template llama2 \
    --finetuning_type lora \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-DBTHub_Prompt-chase-WB-FIO/checkpoint-1250 \
    --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-DBTHub_Prompt-chase-WB-FIO-checkpoint-1250/ \
    --fp16


## CodeLLaMA no-template-no-fp16
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
    --finetuning_type lora \
    --template llama2 \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-DBTHub_Prompt-chase-WB-FIO/checkpoint-1250 \
    --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-DBTHub_Prompt-chase-WB-FIO-checkpoint-1250-nofp16/ 


## CodeLLaMA 13-xds1-qv no-fp16
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/codellama/CodeLlama-13b-Instruct-hf/main \
    --finetuning_type lora \
    --template llama2 \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-20240121_161722-xds1-qv-DBTHub_Prompt-chase-WB-FIO/checkpoint-1200 \
    --output_dir /root/space/models/FineTuned/CodeLlama-13b-Instruct-hf-20240121_161722-xds1-qv-DBTHub_Prompt-chase-WB-FIO-checkpoint-1200-nofp16/ 



## CodeLLaMA 34  spider
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/codellama/CodeLlama-34b-Instruct-hf/main \
    --finetuning_type lora \
    --template llama2 \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-20240120_231353-xds2-qv-DBTHub_Prompt-spider-WB-FIO-UseSQL/checkpoint-1 \
    --output_dir /root/space/models/FineTuned/CodeLlama-34b-Instruct-hf-20240120_231353-xds2-qv-DBTHub_Prompt-spider-WB-FIO-UseSQL-checkpoint-1-nofp16/ 



## CodeLLaMA 34  chase
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/codellama/CodeLlama-34b-Instruct-hf/main \
    --finetuning_type lora \
    --template llama2 \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-20240121_004421-xds2-qv-DBTHub_Prompt-chase-WB-FIO/checkpoint-200 \
    --output_dir /root/space/models/FineTuned/CodeLlama-34b-Instruct-hf-20240121_004421-xds2-qv-DBTHub_Prompt-chase-WB-FIO-checkpoint-200-nofp16/ 


python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/codellama/CodeLlama-34b-Instruct-hf/main \
    --finetuning_type lora \
    --template llama2 \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-20240121_004421-xds2-qv-DBTHub_Prompt-chase-WB-FIO/checkpoint-200 \
    --output_dir /root/space/models/FineTuned/CodeLlama-34b-Instruct-hf-20240121_004421-xds2-qv-DBTHub_Prompt-chase-WB-FIO-checkpoint-200-nofp16/ 