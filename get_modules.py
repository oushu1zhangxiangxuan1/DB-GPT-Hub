from transformers import AutoModel, AutoConfig

# 读入模型路径
# model_path = "/root/space/models/HuggingFaceH4/starchat-beta"
# model_path = "/root/space/models/NumbersStation/nsql-6B"
# model_path = "/root/space/models/bugdaryan/WizardCoderSQL-15B-V1.0/main"
model_path = "/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"

def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear4bit" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    return list(unique_layers)

# 加载模型配置
config = AutoConfig.from_pretrained(model_path)

# 加载模型
model = AutoModel.from_pretrained(model_path, config=config)

# 打印所有modules
print("所有modules:")
print(model)

# 打印可以进行LoRA微调的modules
target_modules = find_target_modules(model)
print("\n可以进行LoRA微调的modules:")
print(target_modules)


# import re
# model_modules = str(model.modules)
# pattern = r'\((\w+)\): Linear'
# linear_layer_names = re.findall(pattern, model_modules)

# names = []
# # Print the names of the Linear layers
# for name in linear_layer_names:
#     names.append(name)
# target_modules = list(set(names))
