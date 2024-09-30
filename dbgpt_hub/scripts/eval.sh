python dbgpt_hub/eval/evaluation.py \
  --input /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/pred/pred_sql.sql \
  --gold /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/spider/dev_gold.sql \
  --db /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/spider/database \
  --table /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/spider/tables.json \
  --etype all \
  --progress_bar_for_each_datapoint







#   --keep_distinct       whether to keep distinct keyword during evaluation. default is false.
#   --natsql              whether to convert natsql to sql and evaluate the converted sql
#   --plug_value          whether to plug in the gold value into the predicted query; suitable if your model does not predict values.