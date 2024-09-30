alias cdut="code /root/space/repos/eosphoros-ai/DB-GPT-Hub/utils/util.sh"
alias mix="conda activate /root/space/conda_envs/mix"
alias v04="conda activate /root/space/conda_envs/v040"
alias v05="conda activate /root/space/conda_envs/v050"
alias v55="conda activate /root/space/conda_envs/v055"
alias tgi="conda activate /root/space/conda_envs/tgi"
# alias inf_conda="c232"
# alias inf_conda="v05"
alias inf_conda="v55"
alias cda=""

get_model_name_by_path() {
    local model_path="$1"
    local model_name=$(basename "$model_path")  # 默认取最后一个路径部分
    
    # 如果最后一个路径部分是"main"，则取倒数第二个
    if [ "$model_name" = "main" ]; then
        model_name=$(basename "$(dirname "$model_path")")
    elif [[ "$model_name" == checkpoint* ]]; then
        model_name=$(basename "$(dirname "$model_path")")-$(echo "$model_name" | cut -d'-' -f2)
    fi
    
    echo "$model_name"
}

# model_path="/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"
# # 调用函数并传入 model_path
# model_name=$(get_model_name_by_path $model_path)
# echo "Model Name: $model_name"


function train_llama {
    model_path="/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"

    max_source_length=12000
    max_target_length=5000
    lora_rank=32
    lora_dropout=0.5

    # max_source_length=4096
    # max_target_length=512
    # lora_rank=16

    lora_alpha=$((lora_rank * 2))
    template='llama2'
    finetuning_type='lora'
    lora_target='q_proj,v_proj'
    learning_rate=2e-4
    streaming=false
    max_steps=200000
    use_quantization=true
    quantization_bit='4'
    quantization_type='fp4'

    local traindata="$(basename $1)"
    echo "traindata: $traindata"

    # 复制文件并替换参数
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/train.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_train.json
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/dev.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_dev.json
    cd /root/space/repos/eosphoros-ai/DB-GPT-Hub/

    streaming_str=''
    if $streaming; then
        streaming_str="--streaming --max_steps=$max_steps"
    fi

    quantization_param="--fp16"
    quantization_name='fp16'
    if $use_quantization; then
        # quantization_param=" --quantization_bit $quantization_bit"
        quantization_param=""
        quantization_name="qb_$quantization_bit"
        if [ $quantization_bit -eq 4 ]; then
            quantization_param+="  --quantization_type=$quantization_type"
            quantization_name+="qt_$quantization_type"
        fi
    fi

    model_name=$(get_model_name_by_path $model_path)
    underscore_lora_target=$(echo "$lora_target" | sed 's/,/_/g')

    output_dir="dbgpt_hub/output/adapter/$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}/"

    local logfile="/root/space/logs/fschat/train_$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}.log"
    echo "logfile: $logfile"

    echo "quantization_param: $quantization_param"

    cth
    export CUDA_VISIBLE_DEVICES=$2
    wandb offline
    nohup python dbgpt_hub/train/sft_train.py \
        --quantization_bit 4 \
        --quantization_type fp4 \
        --model_name_or_path $model_path \
        --do_train \
        --dataset $traindata \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --template $template \
        --finetuning_type $finetuning_type \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_target $lora_target \
        --output_dir $output_dir \
        --overwrite_cache \
        --lora_dropout $lora_dropout \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine_with_restarts \
        --logging_steps 25 \
        --save_steps 50 \
        --learning_rate $learning_rate \
        --num_train_epochs 10 \
        --plot_loss true \
        > $logfile 2>&1 &

    cda
    cd -
    disown
    tail -f $logfile
}


function test_train {
    model_path="/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"

    max_source_length=12000
    max_target_length=5000
    lora_rank=64

    # max_source_length=4096
    # max_target_length=512
    # lora_rank=16

    lora_alpha=$((lora_rank * 2))
    template='llama2'
    finetuning_type='lora'
    lora_target='q_proj,v_proj'
    learning_rate=2e-4
    streaming=false
    max_steps=200000
    use_quantization=true
    quantization_bit=4
    quantization_type='nf4'

    local traindata="$(basename $1)"
    echo "traindata: $traindata"

    # 复制文件并替换参数
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/train.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_train.json
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/dev.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_dev.json
    cd /root/space/repos/eosphoros-ai/DB-GPT-Hub/

    cth

    export CUDA_VISIBLE_DEVICES=$2

    echo "traindata: $traindata"
    streaming_str=''
    if $streaming; then
        streaming_str="--streaming --max_steps=$max_steps"
    fi

    quantization_param='--fp16'
    if $use_quantization; then
        quantization_param="--quantization_bit=$quantization_bit"
        if [ $quantization_bit -eq 4 ]; then
            quantization_param+="  --quantization_type=$quantization_type"
        fi
    fi

    model_name=$(get_model_name_by_path $model_path)
    underscore_lora_target=$(echo "$lora_target" | sed 's/,/_/g')

    output_dir="dbgpt_hub/output/adapter/$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-${traindata}/"

    local logfile="/root/space/logs/fschat/train_$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-${traindata}.log"
    echo "logfile: $logfile"

    echo "quantization_param: $quantization_param"

    wandb offline
    nohup python dbgpt_hub/train/sft_train.py $streaming_str  $quantization_param\
        --model_name_or_path $model_path \
        --do_train \
        --dataset $traindata \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --template $template \
        --finetuning_type $finetuning_type \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_target $lora_target \
        --output_dir $output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine_with_restarts \
        --logging_steps 25 \
        --save_steps 50 \
        --learning_rate $learning_rate \
        --num_train_epochs 10 \
        --quantization_type $quantization_type \
        --plot_loss true \
        > $logfile 2>&1 &

    cda
    cd -
    disown
    tail -f $logfile
}


function ds_codellama {
   model_path="/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"

    max_source_length=12000
    max_target_length=5000
    lora_rank=64
    lora_alpha=$((lora_rank * 2))
    template='llama2'
    finetuning_type='lora'
    lora_target='q_proj,v_proj'
    learning_rate=2e-4
    streaming=false
    max_steps=200000
    use_quantization=false
    quantization_bit=4

    local traindata="$(basename $1)"
    echo "traindata: $traindata"

    # 复制文件并替换参数
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/train.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_train.json
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/dev.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_dev.json
    cd /root/space/repos/eosphoros-ai/DB-GPT-Hub/

    cth

    export CUDA_VISIBLE_DEVICES=$2

    echo "traindata: $traindata"

    streaming_str=''
    if $streaming; then
        streaming_str="--streaming --max_steps=$max_steps"
    fi

    quantization_param='--fp16'
    if $use_quantization; then
        quantization_param="--quantization_bit=$quantization_bit"
        if [ $quantization_bit -eq 4 ]; then
            quantization_param+=" --quantization_type=$quantization_type"
        fi
    fi

    model_name=$(get_model_name_by_path $model_path)
    underscore_lora_target=$(echo "$lora_target" | sed 's/,/_/g')

    output_dir="dbgpt_hub/output/adapter/$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-${traindata}/"

    local logfile="/root/space/logs/fschat/train_$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-${traindata}.log"
    echo "logfile: $logfile"

    wandb offline
    # 定义起始端口号
    master_port=$(shuf -i 20000-30000 -n 1)
    nohup deepspeed \
        --master_port $master_port \
        --include=v1004:$2 \
        --hostfile=/root/space/repos/eosphoros-ai/DB-GPT-Hub/hostfile \
        dbgpt_hub/train/sft_train.py $quantization_param \
        --model_name_or_path $model_path \
        --do_train $streaming_str \
        --dataset $traindata \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --template $template \
        --finetuning_type $finetuning_type \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_target $lora_target \
        --output_dir $output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine_with_restarts \
        --logging_steps 25 \
        --save_steps 50 \
        --learning_rate $learning_rate \
        --num_train_epochs 10 \
        --plot_loss > $logfile 2>&1 &

    cda
    cd -
    disown
    tail -f $logfile
}



function ds_stage {
    model_path="/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"

    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds1.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_of.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_nooffload.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_ofparam.json"
    max_source_length=12000
    max_target_length=5000
    lora_rank=64
    lora_alpha=$((lora_rank * 2))
    template='llama2'
    finetuning_type='lora'
    lora_target='q_proj,v_proj'
    learning_rate=2e-4
    streaming=false
    max_steps=200000
    use_quantization=false
    quantization_bit=4
    quantization_type='nf4'

    local traindata="$(basename $1)"
    echo "traindata: $traindata"

    # 复制文件并替换参数
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/train.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_train.json
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/dev.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_dev.json
    cd /root/space/repos/eosphoros-ai/DB-GPT-Hub/


    streaming_str=''
    if $streaming; then
        streaming_str="--streaming --max_steps=$max_steps"
    fi


    quantization_param="--fp16"
    quantization_name='fp16'
    if $use_quantization; then
        # quantization_param=" --quantization_bit $quantization_bit"
        quantization_param=""
        quantization_name="qb_$quantization_bit"
        if [ $quantization_bit -eq 4 ]; then
            quantization_param+="  --quantization_type=$quantization_type"
            quantization_name+="qt_$quantization_type"
        fi
    fi

    model_name=$(get_model_name_by_path $model_path)
    underscore_lora_target=$(echo "$lora_target" | sed 's/,/_/g')

    output_dir="dbgpt_hub/output/adapter/$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}/"

    local logfile="/root/space/logs/fschat/train_$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}.log"
    echo "logfile: $logfile"
    
    
    hubc12

    export CUDA_VISIBLE_DEVICES=$2
    wandb offline
    # 定义起始端口号
    master_port=$(shuf -i 20000-30000 -n 1)
    nohup deepspeed \
        --master_port $master_port \
        --include=v1004:$2 \
        --hostfile=/root/space/repos/eosphoros-ai/DB-GPT-Hub/hostfile \
        dbgpt_hub/train/sft_train.py $quantization_param \
        --deepspeed $stage_config \
        --model_name_or_path $model_path \
        --do_train $streaming_str \
        --dataset $traindata \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --template $template \
        --finetuning_type $finetuning_type \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_target $lora_target \
        --output_dir $output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine_with_restarts \
        --logging_steps 25 \
        --save_steps 50 \
        --learning_rate $learning_rate \
        --num_train_epochs 10 \
        --plot_loss > $logfile 2>&1 &

    cda
    cd -
    disown
    tail -f $logfile
}


function ds_34_stage_3 {
    model_path="/root/space/models/codellama/CodeLlama-34b-Instruct-hf/main"

    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds1.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_of.json"
    # stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_nooffload.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_ofparam.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_ofparam_sgd.json"
    max_source_length=12000
    max_target_length=5000
    lora_rank=8
    lora_alpha=$((lora_rank * 2))
    template='llama2'
    finetuning_type='lora'
    lora_target='q_proj,v_proj'
    learning_rate=2e-4
    streaming=false
    max_steps=200000
    use_quantization=false
    quantization_bit=4
    quantization_type='nf4'

    num_train_epochs=1000

    local traindata="$(basename $1)"
    echo "traindata: $traindata"

    # 复制文件并替换参数
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/train.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_train.json
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/dev.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_dev.json
    cd /root/space/repos/eosphoros-ai/DB-GPT-Hub/


    streaming_str=''
    if $streaming; then
        streaming_str="--streaming --max_steps=$max_steps"
    fi


    quantization_param="--fp16"
    quantization_name='fp16'
    if $use_quantization; then
        quantization_param=" --quantization_bit $quantization_bit"
        # quantization_param=""
        quantization_name="qb_$quantization_bit"
        if [ $quantization_bit -eq 4 ]; then
            quantization_param+="  --quantization_type $quantization_type"
            quantization_name+="qt_$quantization_type"
        fi
    fi

    model_name=$(get_model_name_by_path $model_path)
    underscore_lora_target=$(echo "$lora_target" | sed 's/,/_/g')

    output_dir="dbgpt_hub/output/adapter/$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}/"

    local logfile="/root/space/logs/fschat/train_$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}.log"
    echo "logfile: $logfile"
    
    
    hubc12
    # cth

    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
    # export PYTORCH_CUDA_ALLOC_CONF=max_cached_bytes:1073741824,max_split_size_mb:32


    export CUDA_VISIBLE_DEVICES=$2
    wandb offline
    # 定义起始端口号
    master_port=$(shuf -i 20000-30000 -n 1)
    nohup deepspeed \
        --master_port $master_port \
        --include=v1004:$2 \
        --hostfile=/root/space/repos/eosphoros-ai/DB-GPT-Hub/hostfile \
        dbgpt_hub/train/sft_train.py $quantization_param \
        --deepspeed $stage_config \
        --model_name_or_path $model_path \
        --do_train $streaming_str \
        --dataset $traindata \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --template $template \
        --finetuning_type $finetuning_type \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_target $lora_target \
        --output_dir $output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine_with_restarts \
        --logging_steps 25 \
        --save_steps 50 \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --plot_loss > $logfile 2>&1 &

    cda
    cd -
    disown
    tail -f $logfile
}


fv(){
    # t2c12
    inf_conda

    # 定义起始端口号
    start_port=$(shuf -i 30000-40000 -n 1)

    # 查找可用端口
    while true; do
        if ! lsof -i :$start_port > /dev/null 2>&1; then
            break
        fi
        start_port=$((start_port + 1))
    done

    local model_path="$1"
    local current_datetime="$(date +'%Y%m%d_%H%M%S')"

    local model_name=$(basename "$model_path")  # 默认取最后一个路径部分
    # 如果最后一个路径部分是"main"，则取倒数第二个
    if [ "$model_name" = "main" ]; then
        model_name=$(basename "$(dirname "$model_path")")
    fi
    
    local log_filename="/root/space/logs/fschat/fc_model_${current_datetime}_${model_name}.log"

    cd /root/space/logs/fschat/
    # local should_deactivate=$(try_activate "c232")
    export CUDA_VISIBLE_DEVICES=$2
    cuda_device_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    # unset CUDA_VISIBLE_DEVICES
    nohup python -m fastchat.serve.vllm_worker \
    --dtype half \
    --model-path $1  \
    --model-name $model_name \
    --host 0.0.0.0 \
    --port $start_port \
    --controller http://localhost:21001 \
    --worker-address http://localhost:$start_port \
    --gpu-memory-utilization 0.8 \
    --load-format auto \
    --tensor-parallel-size $cuda_device_count \
    > "$log_filename" 2>&1 &

    # try_cda "$should_deactivate"
    disown
    cda
    cd -
    tail -f "$log_filename"
}


function ds_13 {
    model_path="/root/space/models/codellama/CodeLlama-13b-Instruct-hf/main"

    # stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds1.json"
    # stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config.json"
    # stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_of.json"
    stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_nooffload.json"
    # stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_ofparam.json"
    # stage_config="/root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/configs/ds_config_stage3_ofparam_sgd.json"
    lora_dropout=0.1
    max_source_length=12000
    max_target_length=5000
    lora_rank=64
    lora_alpha=$((lora_rank * 2))
    template='llama2'
    finetuning_type='lora'
    lora_target='q_proj,v_proj'
    learning_rate=2e-4
    streaming=false
    max_steps=200000

    use_quantization=false
    quantization_bit=4
    quantization_type='nf4'

    num_train_epochs=1000

    local traindata="$(basename $1)"
    echo "traindata: $traindata"

    # 复制文件并替换参数
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/train.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_train.json
    ln -sf /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/finetune/${traindata}/dev.json /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/data/${traindata}_dev.json
    cd /root/space/repos/eosphoros-ai/DB-GPT-Hub/


    streaming_str=''
    if $streaming; then
        streaming_str="--streaming --max_steps=$max_steps"
    fi


    quantization_param="--fp16"
    quantization_name='fp16'
    if $use_quantization; then
        quantization_param=" --quantization_bit $quantization_bit"
        # quantization_param=""
        quantization_name="qb_$quantization_bit"
        if [ $quantization_bit -eq 4 ]; then
            quantization_param+="  --quantization_type $quantization_type"
            quantization_name+="qt_$quantization_type"
        fi
    fi

    model_name=$(get_model_name_by_path $model_path)
    underscore_lora_target=$(echo "$lora_target" | sed 's/,/_/g')

    output_dir="dbgpt_hub/output/adapter/$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}/"

    local logfile="/root/space/logs/fschat/train_$model_name-ds-$(date +"%Y%m%d_%H%M%S")-tpl_$template-src_$max_source_length-tgt_$max_target_length-r_$lora_rank-a_$lora_alpha-lt_$underscore_lora_target-lr_$learning_rate-stream_$streaming-$quantization_name-${traindata}.log"
    echo "logfile: $logfile"
    
    
    hubc12
    # hubc12

    # export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
    # export PYTORCH_CUDA_ALLOC_CONF=max_cached_bytes:1073741824,max_split_size_mb:32
    export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:32


    export CUDA_VISIBLE_DEVICES=$2
    wandb offline
    # 定义起始端口号
    master_port=$(shuf -i 20000-30000 -n 1)

    # --num_gpus 4 
    # TODO: 
    # --repetition_penalty  --lora_dropout
    # --resume_lora_training --resume_from_checkpoint
    # L20 可以支持bp16?

    nohup deepspeed \
        --master_port $master_port \
        --include=v1004:$2 \
        --hostfile=/root/space/repos/eosphoros-ai/DB-GPT-Hub/hostfile \
        dbgpt_hub/train/sft_train.py $quantization_param \
        --deepspeed $stage_config \
        --model_name_or_path $model_path \
        --do_train $streaming_str \
        --dataset $traindata \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --template $template \
        --finetuning_type $finetuning_type \
        --lora_rank $lora_rank \
        --lora_alpha $lora_alpha \
        --lora_target $lora_target \
        --lora_dropout $lora_dropout \
        --output_dir $output_dir \
        --overwrite_cache \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --lr_scheduler_type cosine_with_restarts \
        --logging_steps 5 \
        --save_steps 50 \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --plot_loss > $logfile 2>&1 &

        # --resume_from_checkpoint /root/space/repos/eosphoros-ai/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-13b-Instruct-hf-ds-20240326_201008-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-OushuDBT_Prompt_0322-oushu0326-WB-FIO-UseSQL-SF-FR/checkpoint-210 \


    cda
    cd -
    disown
    tail -f $logfile
}


vl(){
    inf_conda
    block_size=8
    max_num_seqs=1
    max_model_len=20000
    local model_path="$1"
    local current_datetime="$(date +'%Y%m%d_%H%M%S')"

    local model_name=$(basename "$model_path")  # 默认取最后一个路径部分
    # 如果最后一个路径部分是"main"，则取倒数第二个
    if [ "$model_name" = "main" ]; then
        model_name=$(basename "$(dirname "$model_path")")
    fi
    # 如果model_name包含"gguf"（不区分大小写），则将其中的.替换为_
    if [[ "$model_name" == *gguf* ]]; then
        model_name=${model_name//./_}
    fi
    echo "Model name: $model_name"

    
    local log_filename="/root/space/logs/fschat/model_${current_datetime}_${model_name}.log"

    echo $log_filename

    cd /root/space/logs/fschat/
    # local should_deactivate=$(try_activate "c232")
    export CUDA_VISIBLE_DEVICES="$2"
    cuda_device_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


    # echo "praram3 out: $3"
    eLora=""
    if [ -n "$3" ]; then
        # echo "praram3: $3"
        eLora="--enable-lora --max-lora-rank 64 --lora-modules "
        # 比如 $3 = /home/repos/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO/
        # TODO:
        # 列出 $3 中所有的一级子文件夹的绝对路径
        # 对于每个绝对路径 abs_p，取最后两级的目录名称通过_拼接为lora_name
        # 然后将lora_name和abs_p拼接为 $lora_name=$abs_p 例如CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO_checkpoint-800=/home/repos/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO/checkpoint-800
        # 最终把所有的这些凭借到eLora后面
        for folder in "$3"/*; do
            if [ -d "$folder" ]; then
                # 对于每个绝对路径 abs_p，取最后两级的目录名称通过_拼接为lora_name
                folder_name=$(basename "$folder")
                parent_folder=$(basename "$(dirname "$folder")")
                lora_name="${parent_folder}_${folder_name}"
                lora_name=${lora_name//-Instruct-hf/}
                lora_name=${lora_name//checkpoint/ckpt}
                lora_name=${lora_name//-tpl_llama2/}
                lora_name=${lora_name//-stream_false/}
                # 然后将lora_name和abs_p拼接为 $lora_name=$abs_p
                lora_path="$lora_name=$folder"
                # 最终把所有的这些凭借到eLora后面
                eLora="$eLora$lora_path "
            fi
        done
        # echo "eLora: $eLora"
    fi

    eCT=""
    if [ -n "$ct" ]; then
        # echo "praram3: $3"
        eCT=" --chat-template $ct "
    fi

    IFS=' ' read -r -A args_lora <<< "$eLora"
    IFS=' ' read -r -A args_ct   <<< "$eCT"
   
    nohup python -m vllm.entrypoints.openai.api_server ${args_lora[@]}  ${args_ct[@]} \
        --dtype half \
        --port $port \
        --model $1 \
        --served-model-name $model_name \
        --load-format auto \
        --tensor-parallel-size $cuda_device_count \
        --gpu-memory-utilization 0.99 \
        --block-size $block_size \
        --max_num_seqs $max_num_seqs \
        --trust-remote-code \
        --enable-chunked-prefill=False \
        > "$log_filename" 2>&1 &

        
        # --max_model_len $max_model_len \
        # --tokenizer /root/space/models/mlx-community/Meta-Llama-3-70B-Instruct-4bit/main \

        # --disable-log-requests \
        # --max-context-len-to-capture 2048 \
        # --kv-cache-dtype fp8_e5m2 \
    # try_cda "$should_deactivate"
    disown
    cda
    cd -
    tail -f "$log_filename"
}

hl(){
    local port=8000
    if [ -n "$2" ]; then
        port="$2"
    fi
    # 构建 curl 命令字符串
    cmd="curl http://localhost:$port/v1/chat/completions \
        -H \"Content-Type: application/json\" \
        -d '{\"model\": \"$1\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello, \"}]}'"
    
    # 使用 eval 执行命令
    eval "$cmd"
}


h3(){
    local port=8000
    if [ -n "$2" ]; then
        port="$2"
    fi
    # 构建 curl 命令字符串
    cmd="curl http://localhost:$port/v1/chat/completions \
        -H \"Content-Type: application/json\" \
        -d '{\"model\": \"$1\", \"stop_token_ids\" : [128009], \"messages\": [{\"role\": \"user\", \"content\": \"Hello, \"}]}'"
    
    # 使用 eval 执行命令
    eval "$cmd"
}


vt(){
    inf_conda
    block_size=32
    local model_path="$1"
    local current_datetime="$(date +'%Y%m%d_%H%M%S')"

    local model_name=$(basename "$model_path")  # 默认取最后一个路径部分
    # 如果最后一个路径部分是"main"，则取倒数第二个
    if [ "$model_name" = "main" ]; then
        model_name=$(basename "$(dirname "$model_path")")
    fi
    
    local log_filename="/root/space/logs/fschat/model_${current_datetime}_${model_name}.log"

    cd /root/space/logs/fschat/
    # local should_deactivate=$(try_activate "c232")
    export CUDA_VISIBLE_DEVICES="$2"
    cuda_device_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


    # echo "praram3 out: $3"
    eLora=""
    if [ -n "$3" ]; then
        # echo "praram3: $3"
        eLora="--enable-lora --max-lora-rank 64 --lora-modules "
        # 比如 $3 = /home/repos/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO/
        # TODO:
        # 列出 $3 中所有的一级子文件夹的绝对路径
        # 对于每个绝对路径 abs_p，取最后两级的目录名称通过_拼接为lora_name
        # 然后将lora_name和abs_p拼接为 $lora_name=$abs_p 例如CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO_checkpoint-800=/home/repos/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO/checkpoint-800
        # 最终把所有的这些凭借到eLora后面
        for folder in "$3"/*; do
            if [ -d "$folder" ]; then
                # 对于每个绝对路径 abs_p，取最后两级的目录名称通过_拼接为lora_name
                folder_name=$(basename "$folder")
                parent_folder=$(basename "$(dirname "$folder")")
                lora_name="${parent_folder}_${folder_name}"
                lora_name=${lora_name//-Instruct-hf/}
                lora_name=${lora_name//checkpoint/ckpt}
                lora_name=${lora_name//-tpl_llama2/}
                lora_name=${lora_name//-stream_false/}
                # 然后将lora_name和abs_p拼接为 $lora_name=$abs_p
                lora_path="$lora_name=$folder"
                # 最终把所有的这些凭借到eLora后面
                eLora="$eLora$lora_path "
            fi
        done
        # echo "eLora: $eLora"
    fi

    
    eCT=""
    if [ -n "$ct" ]; then
        # echo "praram3: $3"
        eCT=" --chat-template $ct "
    fi

    IFS=' ' read -r -A args_lora <<< "$eLora"
    IFS=' ' read -r -A args_ct   <<< "$eCT"
   
    nohup python -m vllm.entrypoints.openai.api_server ${args_lora[@]}  ${args_ct[@]} \
        --dtype half \
        --port $port \
        --model $1 \
        --served-model-name $model_name \
        --load-format auto \
        --tensor-parallel-size $cuda_device_count \
        --gpu-memory-utilization 0.95 \
        --block-size $block_size \
        > "$log_filename" 2>&1 &


        # --disable-log-requests \
        # --max-context-len-to-capture 2048 \
        # --kv-cache-dtype fp8_e5m2 \
    # try_cda "$should_deactivate"
    disown
    cda
    cd -
    tail -f "$log_filename"
}

dcz(){
    docker exec -it $1 /bin/zsh
}


v8(){
    inf_conda
    block_size=32
    local model_path="$1"
    local current_datetime="$(date +'%Y%m%d_%H%M%S')"

    local model_name=$(basename "$model_path")  # 默认取最后一个路径部分
    # 如果最后一个路径部分是"main"，则取倒数第二个
    if [ "$model_name" = "main" ]; then
        model_name=$(basename "$(dirname "$model_path")")
    fi
    
    local log_filename="/root/space/logs/fschat/model_${current_datetime}_${model_name}.log"

    cd /root/space/logs/fschat/
    # local should_deactivate=$(try_activate "c232")
    export CUDA_VISIBLE_DEVICES="$2"
    cuda_device_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


    # echo "praram3 out: $3"
    eLora=""
    if [ -n "$3" ]; then
        # echo "praram3: $3"
        eLora="--enable-lora --max-lora-rank 64 --lora-modules "
        # 比如 $3 = /home/repos/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO/
        # TODO:
        # 列出 $3 中所有的一级子文件夹的绝对路径
        # 对于每个绝对路径 abs_p，取最后两级的目录名称通过_拼接为lora_name
        # 然后将lora_name和abs_p拼接为 $lora_name=$abs_p 例如CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO_checkpoint-800=/home/repos/DB-GPT-Hub/dbgpt_hub/output/adapter/CodeLlama-34b-Instruct-hf-ds-20240316_105643-tpl_llama2-src_12000-tgt_5000-r_64-a_128-lt_q_proj_v_proj-lr_2e-4-stream_false-fp16-DBTHub_Prompt-chase-WB-FIO/checkpoint-800
        # 最终把所有的这些凭借到eLora后面
        for folder in "$3"/*; do
            if [ -d "$folder" ]; then
                # 对于每个绝对路径 abs_p，取最后两级的目录名称通过_拼接为lora_name
                folder_name=$(basename "$folder")
                parent_folder=$(basename "$(dirname "$folder")")
                lora_name="${parent_folder}_${folder_name}"
                lora_name=${lora_name//-Instruct-hf/}
                lora_name=${lora_name//checkpoint/ckpt}
                lora_name=${lora_name//-tpl_llama2/}
                lora_name=${lora_name//-stream_false/}
                # 然后将lora_name和abs_p拼接为 $lora_name=$abs_p
                lora_path="$lora_name=$folder"
                # 最终把所有的这些凭借到eLora后面
                eLora="$eLora$lora_path "
            fi
        done
        # echo "eLora: $eLora"
    fi

    
    eCT=""
    if [ -n "$ct" ]; then
        # echo "praram3: $3"
        eCT=" --chat-template $ct "
    fi

    IFS=' ' read -r -A args_lora <<< "$eLora"
    IFS=' ' read -r -A args_ct   <<< "$eCT"
   
    nohup python -m vllm.entrypoints.openai.api_server ${args_lora[@]}  ${args_ct[@]} \
        --dtype auto \
        --port $port \
        --model $1 \
        --served-model-name $model_name \
        --load-format auto \
        --tensor-parallel-size $cuda_device_count \
        --gpu-memory-utilization 0.95 \
        --block-size $block_size \
        --swap-space 25 \
        > "$log_filename" 2>&1 &


        # --disable-log-requests \
        # --max-context-len-to-capture 2048 \
        # --kv-cache-dtype fp8_e5m2 \
    # try_cda "$should_deactivate"
    disown
    cda
    cd -
    tail -f "$log_filename"
}