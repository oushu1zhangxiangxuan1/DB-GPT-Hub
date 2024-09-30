from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
import os
from accelerate import init_empty_weights

# 定义函数，找到可以进行LoRA微调的modules
def find_target_modules(model):
    unique_layers = set()
    for name, module in model.named_modules():
        if "Linear4bit" in str(type(module)):
            layer_type = name.split('.')[-1]
            unique_layers.add(layer_type)
    return list(unique_layers)

# 定义函数，根据模型路径获取输出文件名
def get_output_filename(model_path):
    if model_path.endswith("main"):
        # 如果路径以"main"结尾，则使用路径名称前的部分作为文件名
        filename = os.path.basename(os.path.dirname(model_path)) + ".txt"
    else:
        # 否则，使用路径的最后一个部分作为文件名
        filename = os.path.basename(model_path) + ".txt"
    return filename


def main(model_paths):
    # 遍历模型路径
    for model_path in model_paths:
        # 加载模型配置
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # 加载模型
        model = None
        with init_empty_weights():
            # model = AutoModel.from_pretrained(model_path, config=config)
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        # 获取可以进行LoRA微调的modules
        target_modules = find_target_modules(model)

        # 打印所有modules
        print(f"模型路径: {model_path}")
        print("所有modules:")
        print(model)

        # 打印可以进行LoRA微调的modules
        print("\n可以进行LoRA微调的modules:")
        print(target_modules)

        # 获取输出文件名
        output_filename = get_output_filename(model_path)

        # 写入结果到文件
        with open(f"model_modules/{output_filename}", "w") as file:
            file.write(f"模型路径: {model_path}\n")
            file.write("所有modules:\n")
            file.write(str(model))
            file.write("\n\n可以进行LoRA微调的modules:\n")
            file.write(str(target_modules))

        print(f"结果已写入文件: model_modules/{output_filename}\n")
        del model
        del config


if '__main__' == __name__:

    # 模型路径列表
    model_paths = [
        # "/root/space/models/HuggingFaceH4/starchat-beta",
        # "/root/space/models/NumbersStation/nsql-6B",
        # "/root/space/models/bugdaryan/WizardCoderSQL-15B-V1.0/main",
        # "/root/space/models/BAAI/AquilaCode-multi/main",
        "/root/space/models/BAAI/AquilaChat2-7B-16K/main",
        # "/root/space/models/BAAI/AquilaChat2-34B/main",
        # "/root/space/models/Skywork/Skywork-13B-base/main",
        "/root/space/models/WizardLM/WizardCoder-15B-V1.0/main",
        "/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"
    ]


    # model_paths = [
    #     # "/root/space/models/HuggingFaceH4/starchat-beta",
    #     # "/root/space/models/NumbersStation/nsql-6B",
    #     # "/root/space/models/bugdaryan/WizardCoderSQL-15B-V1.0/main",
    #     # "/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"
    #     # "/root/space/models/BAAI/AquilaCode-multi/main",
    #     # "/root/space/models/BAAI/AquilaChat2-7B-16K/main",
    #     # "/root/space/models/BAAI/AquilaChat2-34B/main",
    #     "/root/space/models/Skywork/Skywork-13B-base/main",
    #     # "/root/space/models/WizardLM/WizardCoder-15B-V1.0/main",
    # ]

    main(model_paths)