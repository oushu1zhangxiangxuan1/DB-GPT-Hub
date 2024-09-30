import os
import json
import jsonlines
import sys
import sqlite3
import copy

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from tqdm import tqdm

from dbgpt_hub.configs.config import (
    SQL_DATA_INFO,
    DATA_PATH,
    INPUT_PROMPT,
    INSTRUCTION_PROMPT,
    FIRST_INSTRUCTION_ONLY,
    METADATA_USE_SQL,
    WITH_BACKTICKS,
    FULL_HISTORY,
    PROMPT_NAME,
    SHUFFLE,
    FULL_ROUND,
    USE_DIALECT,
)

def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.sqlite'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return file_paths

def extract_table_sql(db_file):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    table_sql = {}
    for table in tables:
        table_name = table["name"]
        sql = table["sql"]
        table_sql[table_name] = sql

    conn.close()
    return table_sql


def process_sqlite_files(file_list):
    database_tables = {}
    for db_file in file_list:
        if os.path.isfile(db_file) and db_file.endswith(".sqlite"):
            db_name = os.path.splitext(os.path.basename(db_file))[0]
            # print(db_name)
            table_sql = extract_table_sql(db_file)
            database_tables[db_name] = table_sql

    return database_tables

class ProcessSqlData:
    def __init__(self, train_file=None, dev_file=None, use_sql=False) -> None:
        self.train_file = train_file
        self.dev_file = dev_file
        self.use_sql = use_sql

    def decode_json_file(self, data_file_list, table_file, db_id_name, is_multiple_turn=False, database_dir=None, data_info=None):
        """
        TO DO:
            1.将相关prompt放入config中
            2.将不同数据来源的字段信息放入config中
        """

        if table_file.endswith(".jsonl"):
            tables = jsonlines.open(table_file)
            datas = []
            for data_file in data_file_list:
                datas.extend(jsonlines.open(data_file))

        elif table_file.endswith(".json"):
            tables = json.load(open(table_file))
            datas = []
            for data_file in data_file_list:
                datas.extend(json.load(open(data_file)))
        else:
            print("Unsupported file types")
            raise

        # 先将db_id 的table和coloumns处理好
        db_dict = {}
        if "meta_file" in data_info.keys():
            meta_file = data_info["meta_file"]
            meta_file_path = os.path.join(DATA_PATH, data_info["data_source"], meta_file)
            meta_data = json.load(open(meta_file_path))
            for key, db in meta_data.items():
                combined_value=''
                for _, value in db.items():
                    combined_value+=value.strip()
                    combined_value+='\n'
                db_dict[key] = combined_value
        elif self.use_sql:
            files = get_all_file_paths(DATA_PATH)
            dbs = process_sqlite_files(files)
            for key, value in dbs.items():
                combined_value = ";\n".join(value.values()) + ";"
                db_dict[key] = combined_value
        else:
            for item in tables:
                tables = item["table_names_original"]
                coloumns = item["column_names_original"][1:]
                # primary_key = item["primary_keys"]
                # foreign_keys = item["foreign_keys"]
                source = (
                    item["db_id"] + " contains tables such as " + ", ".join(tables) + ". "
                )
                for i, name in enumerate(tables):
                    data = [coloumn[1] for coloumn in coloumns if coloumn[0] == i]
                    source += (
                        "Table " + name + " has columns such as " + ", ".join(data) + ". "
                    )

                #     # get primary key info
                #     for j in range(len(primary_key)):
                #         if coloumns[primary_key[j] - 1][0] == i:
                #             source += (
                #                 coloumns[primary_key[j] - 1][1]
                #                 + " is the primary key."
                #                 + "\n"
                #             )

                # # get foreign key info
                # for key in foreign_keys:
                #     source += (
                #         "The "
                #         + coloumns[key[0] - 1][1]
                #         + " of "
                #         + tables[coloumns[key[0] - 1][0]]
                #         + " is the foreign key of "
                #         + coloumns[key[1] - 1][1]
                #         + " of "
                #         + tables[coloumns[key[1] - 1][0]]
                #         + ".\n"
                #     )

                db_dict[item["db_id"]] = source

        # 单论对话
        res = []
        for data in tqdm(datas):
            if data[db_id_name] in db_dict.keys():
                if USE_DIALECT and "dialect" in data_info:
                    sub_INSTRUCTION_PROMPT = INSTRUCTION_PROMPT.format(data_info["dialect"], '{}', '{}', '{}')
                instruction = sub_INSTRUCTION_PROMPT.format(db_dict[data[db_id_name]])
                if is_multiple_turn:
                    first = True
                    history = []
                    for i, interaction in enumerate(data["interaction"]):
                        if FIRST_INSTRUCTION_ONLY and not first:
                            instruction = ""
                            first = False
                        
                        output = interaction["query"]
                        if WITH_BACKTICKS:
                            output = """```sql\n{}\n```""".format(output)
            
                        input = {
                            "db_id": data[db_id_name],
                            "instruction": instruction,
                            "input": INPUT_PROMPT.format(interaction["utterance"]),
                            "output": output,
                            "history": history if FULL_HISTORY else copy.deepcopy(history),
                        }
                        if not FULL_ROUND:
                            res.append(input)
                        else:
                            if i+1==len(data["interaction"]):
                                res.append(input)
                        history.append((INPUT_PROMPT.format(interaction["utterance"]), output))
                else:
                    output = data["query"]
                    if WITH_BACKTICKS:
                        output = """```sql\n{}\n```""".format(output)
                    input = {
                        "db_id": data[db_id_name],
                        "instruction": instruction,
                        "input": INPUT_PROMPT.format(data["question"]),
                        "output": output,
                        "history": [],
                    }
                    res.append(input)
        return res

    def create_sft_raw_data(self):
        train_data = []
        dev_data = []
        for data_info in SQL_DATA_INFO:
            train_data_file_list = [
                os.path.join(DATA_PATH, data_info["data_source"], file)
                for file in data_info["train_file"]
            ]
            train_data.extend(
                self.decode_json_file(
                    data_file_list=train_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH, data_info["data_source"], data_info["tables_file"]
                    ),
                    db_id_name=data_info["db_id_name"],
                    is_multiple_turn=data_info['is_multiple_turn'],
                    database_dir=os.path.join(
                        DATA_PATH, data_info["data_source"],"database"
                    ),
                    data_info=data_info,
                )
            )

            dev_data_file_list = [
                os.path.join(DATA_PATH, data_info["data_source"], file)
                for file in data_info["dev_file"]
            ]
            dev_data.extend(
                self.decode_json_file(
                    data_file_list=dev_data_file_list,
                    table_file=os.path.join(
                        DATA_PATH, data_info["data_source"], data_info["tables_file"]
                    ),
                    db_id_name=data_info["db_id_name"],
                    is_multiple_turn=data_info['is_multiple_turn'],
                    database_dir=os.path.join(
                        DATA_PATH, data_info["data_source"],"database"
                    ),
                    data_info=data_info,
                )
            )
        with open(self.train_file, "w", encoding="utf-8") as s:
            if SHUFFLE:
                import random
                random.shuffle(train_data)
            json.dump(train_data, s, indent=4, ensure_ascii=False)
        with open(self.dev_file, "w", encoding="utf-8") as s:
            json.dump(dev_data, s, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # FIO: FirstInstructionOnly
    # WB: WithBackticks
    # prompt = "MetadataPrompt_1"
    prompt = PROMPT_NAME
    # prompt = "DBTHub_Prompt_CH"

    if len(SQL_DATA_INFO)==0:
        raise "SQL_DATA_INFO config is null!"
    data_prefixs = []
    for data_info in SQL_DATA_INFO:
        data_prefixs.append(data_info["data_source"])
    prompt += ("-"+"_".join(data_prefixs))

    if WITH_BACKTICKS:
        prompt += "-WB"
    if FIRST_INSTRUCTION_ONLY:
        prompt += "-FIO"
    if METADATA_USE_SQL:
        prompt += "-UseSQL"
    if FULL_HISTORY:
        prompt += "-FH"
    if SHUFFLE:
        prompt += "-SF"
    if FULL_ROUND:
        prompt += "-FR"
    

    # dbgpt_hub/data/finetune/MetadataPrompt_1_1

    print(prompt)
    
    output_path = os.path.join(DATA_PATH, "finetune", prompt)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_in_one_train_file = os.path.join(output_path, "train.json")
    all_in_one_dev_file = os.path.join(output_path, "dev.json")
    precess = ProcessSqlData(
        train_file=all_in_one_train_file, 
        dev_file=all_in_one_dev_file,
        use_sql=METADATA_USE_SQL,
        )
    precess.create_sft_raw_data()
