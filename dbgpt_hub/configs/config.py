import os

### path config
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ROOT_PATH = "/root/autodl-tmp"
# MODELS_PARENT_PATH = "/home/model_files/codellama/"
# DEFAULT_FT_MODEL_NAME = "CodeLlama-7b-Instruct-hf"
MODELS_PARENT_PATH = "/home/model/"
DEFAULT_FT_MODEL_NAME = "Baichuan2-13B-Chat"
MODEL_PATH = os.path.join(MODELS_PARENT_PATH, DEFAULT_FT_MODEL_NAME)

# MODEL_PATH = os.path.join(ROOT_PATH, "model")
ADAPTER_PATH = os.path.join(ROOT_PATH, "dbgpt_hub/output/adapter")
MERGED_MODELS = os.path.join(ROOT_PATH, "dbgpt_hub/output/merged_models")

# DATA_PATH = "/root/autodl-tmp/data/spider/pre_processed_data"
# OUT_DIR= "/root/autodl-tmp/codellama"

DATA_PATH = os.path.join(ROOT_PATH, "dbgpt_hub/data")
PREDICTED_DATA_PATH = os.path.join(ROOT_PATH, "dbgpt_hub/data/eval_data/dev_sql.json")
PREDICTED_OUT_FILENAME = "pred_sql.sql"
# OUT_DIR = os.path.join(DATA_PATH, "out_pred")
OUT_DIR = os.path.join(ROOT_PATH, "dbgpt_hub/output/")

## model constants
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


LOG_FILE_NAME = "trainer_log.jsonl"

# head_state_dict,model save name
VALUE_HEAD_FILE_NAME = "value_head.bin"

# output ,finetuning_args save_to_json name
FINETUNING_ARGS_NAME = "finetuning_args.json"

#  when prepare_model_for_training ,layer_norm_names
LAYERNORM_NAMES = ["norm", "ln_f", "ln_attn", "ln_mlp"]
EXT2TYPE = {"csv": "csv", "json": "json", "jsonl": "json", "txt": "text"}

# text2sql dataset information for processing sql data
# TODO: BIRD \ WiKiSQL \ ...
SQL_DATA_INFO = [
    {
        "data_source": "spider",
        "train_file": ["train_spider.json", "train_others.json"],
        "dev_file": ["dev.json"],
        "tables_file": "tables.json",
        "db_id_name": "db_id",
        "is_multiple_turn": False,
        "dialect":"sqlite",
    }
    ,
    # {
    #     "data_source": "chase",
    #     "train_file": ["Chase/chase_train.json"],
    #     "dev_file": ["Chase/chase_dev.json"],
    #     "tables_file": "Chase/chase_tables.json",
    #     "db_id_name": "database_id",
    #     "is_multiple_turn": True,
    # }
    # ,
    # {
    #     "data_source": "oushu0322",
    #     "train_file": ["all.json", "dev.json"],
    #     "dev_file": ["all.json"],
    #     "meta_file": "meta.json",
    #     "tables_file": "table.json",
    #     "db_id_name": "database_id",
    #     "is_multiple_turn": True,
        # "dialect":"postgresql",
    # }
    # ,
    {
        "data_source": "oushu0326",
        "train_file": ["train.json"],
        "dev_file": ["dev.json"],
        "meta_file": "meta.json",
        "tables_file": "table.json",
        "db_id_name": "database_id",
        "is_multiple_turn": True,
        "dialect":"postgresql",
    }
    ,
    {
        "data_source": "chase",
        "train_file": ["Chase/chase_train.json"],
        "dev_file": ["Chase/chase_dev.json"],
        "tables_file": "Chase/chase_tables.json",
        "db_id_name": "database_id",
        "is_multiple_turn": True,
        "dialect":"sqlite",
    }
    # ,
    # {
    #     "data_source": "cosql",
    #     "train_file": ["sql_state_tracking/cosql_train.json"],
    #     "dev_file": ["sql_state_tracking/cosql_dev.json"],
    #     "tables_file": "tables.json",
    #     "db_id_name": "database_id",
    #     "is_multiple_turn": True,
    # }
    # ,
    # {
    #     "data_source": "sparc",
    #     "train_file": ["train.json"],
    #     "dev_file": ["dev.json"],
    #     "tables_file": "tables.json",
    #     "db_id_name": "database_id",
    #     "is_multiple_turn": True,
    # }
]

SHUFFLE=True
FIRST_INSTRUCTION_ONLY = True
FULL_ROUND=False
METADATA_USE_SQL = True
WITH_BACKTICKS = True
FULL_HISTORY = False  # each turn will has a full history When set to True.(only works on multiturn datasets)
USE_DIALECT = True

# PROMPT_NAME = "MetadataPrompt_1"
# PROMPT_NAME = "DBTHub_Prompt_CH"


# INSTRUCTION_PROMPT = """
# ```sql\n{}\n```\n根据上面的几个表结构，在后续的对话中将自然语言查询转换为SQL语句，仅以markdown形式输出SQL，不要生成其他无关内容，且确保SQL是完整且正确的
# """
# INPUT_PROMPT = "结合上下文，将下面自然语言查询转换为完整的SQL语句，仅以markdown形式输出SQL，不要生成其他无关内容，且确保SQL是完整且正确的:\n{}"

# INSTRUCTION_PROMPT = """
# 根据下面的几个表结构，在后续的对话中将自然语言查询转换为SQL语句，仅以markdown形式输出SQL，不要生成其他无关内容，且确保SQL是完整且正确的\n
# ### Instruction:```sql\n{}\n```\n
# """
# INPUT_PROMPT = "###Input:\n{}\n###Response:"

#########################################################
# PROMPT_NAME = "DBTHub_Prompt"
# INSTRUCTION_PROMPT = """\
# I want you to act as a SQL terminal in front of an example database, \
# you need only to return the sql command to me.Below is an instruction that describes a task, \
# Write a response that appropriately completes the request.\n"
# ##Instruction:\n{}\n"""
# INPUT_PROMPT = "###Input:\n{}\n\n###Response:"

##################################################
# PROMPT_NAME = "OushuDBT_Prompt_0322"
# INSTRUCTION_PROMPT = """
# Given an input question, create a syntactically correct postgresql sql.

# Only use the following tables schema to generate sql:
# {}


# Do not query for tables and columns that do not exist. Also, pay attention to which column is in which table.
# """

# INPUT_PROMPT = """
# Question:
# {}


# Think step by step.

# Please respond in Chinese and without any explanation.

# And SQL statements should be formatted using Markdown syntax start with ```sql
# """
#################################################
##################################################
PROMPT_NAME = "OushuDBT_Prompt_0322"
INSTRUCTION_PROMPT = """
Given an input question, create a syntactically correct {} sql.

Only use the following tables schema to generate sql:
{}


Do not query for tables and columns that do not exist. Also, pay attention to which column is in which table.
"""

INPUT_PROMPT = """
Question:
{}


Think step by step.

Please respond in Chinese and without any explanation.

And SQL statements should be formatted using Markdown syntax start with ```sql
"""
#################################################
# PROMPT_NAME = "NULL_Prompt_Inst"

# INSTRUCTION_PROMPT = """##Instruction:\n{}"""

# INPUT_PROMPT = "###Input:\n{}\n\n###Response:"


# PROMPT_NAME = "DBT_Prompt"

# INSTRUCTION_PROMPT = """You are an assistant that answers user specialized database questions.
# Given an input question, create a syntactically correct sqlite sql.

# Only use the following tables schema to generate sql:
# {}
# """

# INPUT_PROMPT = """
# Question:
# {}

# Think step by step.
# Please respond in Chinese and without any explanation.
# And SQL statements should be formatted using Markdown syntax start with ```sql
# """


# METHODS = ["full", "freeze", "lora"]

# STAGES = ["SFT", "Reward Modeling", "PPO", "DPO", "Pre-Training"]

# DATASET_STAGE_MAP = {
#     "SFT": "sft",
#     "Pre-Training": "pt",
#     "Reward Modeling": "rm",
#     "PPO": "sft",
#     "DPO": "rm",
# }

# SUPPORTED_MODELS = {
#     "LLaMA-7B": "huggyllama/llama-7b",
#     "LLaMA-13B": "huggyllama/llama-13b",
#     "LLaMA-30B": "huggyllama/llama-30b",
#     "LLaMA-65B": "huggyllama/llama-65b",
#     "LLaMA2-7B": "meta-llama/Llama-2-7b-hf",
#     "LLaMA2-13B": "meta-llama/Llama-2-13b-hf",
#     "LLaMA2-70B": "meta-llama/Llama-2-70b-hf",
#     "LLaMA2-7B-Chat": "meta-llama/Llama-2-7b-chat-hf",
#     "LLaMA2-13B-Chat": "meta-llama/Llama-2-13b-chat-hf",
#     "LLaMA2-70B-Chat": "meta-llama/Llama-2-70b-chat-hf",
#     "ChineseLLaMA2-7B": "ziqingyang/chinese-llama-2-7b",
#     "ChineseLLaMA2-13B": "ziqingyang/chinese-llama-2-13b",
#     "ChineseLLaMA2-7B-Chat": "ziqingyang/chinese-alpaca-2-7b",
#     "ChineseLLaMA2-13B-Chat": "ziqingyang/chinese-alpaca-2-13b",
#     "BLOOM-560M": "bigscience/bloom-560m",
#     "BLOOM-3B": "bigscience/bloom-3b",
#     "BLOOM-7B1": "bigscience/bloom-7b1",
#     "BLOOMZ-560M": "bigscience/bloomz-560m",
#     "BLOOMZ-3B": "bigscience/bloomz-3b",
#     "BLOOMZ-7B1-mt": "bigscience/bloomz-7b1-mt",
#     "Falcon-7B": "tiiuae/falcon-7b",
#     "Falcon-7B-Chat": "tiiuae/falcon-7b-instruct",
#     "Falcon-40B": "tiiuae/falcon-40b",
#     "Falcon-40B-Chat": "tiiuae/falcon-40b-instruct",
#     "Baichuan-7B": "baichuan-inc/Baichuan-7B",
#     "Baichuan-13B": "baichuan-inc/Baichuan-13B-Base",
#     "Baichuan-13B-Chat": "baichuan-inc/Baichuan-13B-Chat",
#     "Baichuan2-7B": "baichuan-inc/Baichuan2-7B-Base",
#     "Baichuan2-13B": "baichuan-inc/Baichuan2-13B-Base",
#     "Baichuan2-7B-Chat": "baichuan-inc/Baichuan2-7B-Chat",
#     "Baichuan2-13B-Chat": "baichuan-inc/Baichuan2-13B-Chat",
#     "InternLM-7B": "internlm/internlm-7b",
#     "InternLM-7B-Chat": "internlm/internlm-chat-7b",
#     "Qwen-7B": "Qwen/Qwen-7B",
#     "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
#     "XVERSE-13B": "xverse/XVERSE-13B",
#     "ChatGLM2-6B-Chat": "THUDM/chatglm2-6b",
# }

# DEFAULT_MODULE = {
#     "LLaMA": "q_proj,v_proj",
#     "LLaMA2": "q_proj,v_proj",
#     "ChineseLLaMA2": "q_proj,v_proj",
#     "BLOOM": "query_key_value",
#     "BLOOMZ": "query_key_value",
#     "Falcon": "query_key_value",
#     "Baichuan": "W_pack",
#     "Baichuan2": "W_pack",
#     "InternLM": "q_proj,v_proj",
#     "Qwen": "c_attn",
#     "XVERSE": "q_proj,v_proj",
#     "ChatGLM2": "query_key_value",
# }

# DEFAULT_TEMPLATE = {
#     "LLaMA2": "llama2",
#     "ChineseLLaMA2": "llama2_zh",
#     "Baichuan": "baichuan",
#     "Baichuan2": "baichuan2",
#     "InternLM": "intern",
#     "Qwen": "chatml",
#     "ChatGLM2": "chatglm2",
# }
