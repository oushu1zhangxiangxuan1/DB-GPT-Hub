# llama2 series
# python dbgpt_hub/train/export_model.py \
#     --model_name_or_path /home/LLM/CodeLlama-13b-Instruct-hf \
#     --template llama2 \
#     --finetuning_type lora \
#     --checkpoint_dir dbgpt_hub/output/adapter/CodeLlama-13b-sql-lora \
#     --output_dir dbgpt_hub/output/codellama-13b-sql-sft \
#     --fp16


## Baichuan2
python dbgpt_hub/train/export_model.py \
    --model_name_or_path /root/space/models/baichuan-inc/Baichuan2-13B-Chat/main \
    --template baichuan2 \
    --finetuning_type lora \
    --checkpoint_dir /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/Baichuan2-13B-MetadataPrompt_1-Chase-WB-UseSQL-FIO/checkpoint-50 \
    --output_dir /root/space/models/FineTuned/Baichuan2-13B-MetadataPrompt_1-Chase-WB-UseSQL-FIO/checkpoint-50-$(date "+%Y%m%d%H%M%S")/ \
    --fp16
#     # --bf16