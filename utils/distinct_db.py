import json

file_path = "/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/chase/Chase/chase_dev.json"
file_path = "/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/spider/dev.json"

# 从文件中加载JSON数据
with open(file_path, "r") as file:
    data = json.load(file)

# 使用集合(set)来存储不同的database_id值
unique_database_ids = set()

# 遍历数据结构并将不同的database_id值添加到集合中
for item in data:
    if "database_id" in item:
        unique_database_ids.add(item["database_id"])
    elif "db_id" in item:
        unique_database_ids.add(item["db_id"])

# 输出不同database_id值的数量
print("不同的database_id值数量:", len(unique_database_ids))
