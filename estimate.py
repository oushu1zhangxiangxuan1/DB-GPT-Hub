from transformers import AutoModel
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live, estimate_zero2_model_states_mem_needs_all_cold
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live, estimate_zero3_model_states_mem_needs_all_cold
import datetime
from itertools import product


def main():
    # 加载模型
    model = AutoModel.from_pretrained("/root/space/models/codellama/CodeLlama-34b-Instruct-hf/main")
    num_gpus_per_node_list = [2, 3, 4]
    num_nodes_list = [1, 2, 4]
    additional_buffer_factor_list = [1.0, 1.5]

    # 计算笛卡尔积
    cartesian_product = list(product(num_gpus_per_node_list, num_nodes_list, additional_buffer_factor_list))

    # 输出每个组合的值
    for combination in cartesian_product:
        num_gpus, num_nodes, buffer_factor = combination
        print(f"num_gpus: {num_gpus}, num_nodes: {num_nodes}, buffer_factor: {buffer_factor}\n\n")

        print('estimate_zero2_model_states_mem_needs_all_live\n')
        estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=num_gpus, num_nodes=num_nodes, additional_buffer_factor=buffer_factor)

        print('estimate_zero3_model_states_mem_needs_all_live\n')
        estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=num_gpus, num_nodes=num_nodes, additional_buffer_factor=buffer_factor)

        # print('estimate_zero2_model_states_mem_needs_all_cold\n')
        # estimate_zero2_model_states_mem_needs_all_cold(model, num_gpus_per_node=num_gpus, num_nodes=num_nodes, additional_buffer_factor=buffer_factor)

        # print('estimate_zero3_model_states_mem_needs_all_cold\n')
        # estimate_zero3_model_states_mem_needs_all_cold(model, num_gpus_per_node=num_gpus, num_nodes=num_nodes, additional_buffer_factor=buffer_factor)
        print('-------------------------------------------------------------------------------------------------------------\n\n\n\n')


if '__main__'==__name__:
    main()