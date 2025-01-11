### Control LLM: Enhancing Large Language Models Without Catastrophic Forgetting

### Introduction

Large Language Models (LLMs) demand significant computational resources, making it essential to enhance their capabilities without retraining from scratch. A key challenge in this domain is *catastrophic forgetting* (CF), which hampers performance during Continuous Pre-training (CPT) and Continuous Supervised Fine-Tuning (CSFT).

We propose **Control LLM**, a novel approach that leverages parallel pre-trained and expanded transformer blocks, aligning their hidden states through interpolation strategies. This method effectively preserves performance on existing tasks while seamlessly integrating new knowledge.

#### Key Contributions

1. **Performance Improvement**: 
   - **Mathematical Reasoning**: Achieves significant improvements, e.g., +14.4% on Math-Hard.
   - **Coding Performance**: Improves coding capabilities by +10% on MBPP-PLUS.
   - **Multilingual Capabilities**: Enhances multilingual benchmarks, including:
     - +10.6% on C-Eval.
     - +6.8% on CMMLU.
     - +30.2% on CMMLU-0shot-CoT.

2. **State-of-the-Art Results**: 
   - Surpasses existing methods and achieves SOTA among open-source models tuned from the same base model, using substantially less data and compute.
   - Maintains strong original capabilities with minimal degradation (<4.3% on MMLU) compared to >35% observed in other open-source Math and Coding models.

3. **Real-World Deployment**: Successfully deployed in LinkedIn's GenAI-powered job seeker and Ads unit products.

#### Benefits of Control LLM

- Preserves performance on existing tasks while integrating new knowledge.
- Reduces catastrophic forgetting during continuous training and fine-tuning.
- Enhances LLM performance across a variety of domains, including reasoning, coding, and multilingual tasks.

#### Code and Models

To support further research and collaboration, we release:

- **Training and Evaluation Codebase**: [GitHub Repository](https://github.com/linkedin/ControlLLM)
- **Trained Models**: [Hugging Face Models](https://huggingface.co/ControlLLM)

We invite the community to explore, experiment, and contribute to advancing LLM research.

---

Feel free to reach out with feedback, suggestions, or contributions via the [GitHub Issues](https://github.com/linkedin/ControlLLM/issues) section.

### Control LLM model performance
![control_llm_sota_comparison](https://github.com/user-attachments/assets/0f812d8a-ca2f-458b-a5bb-727cdf916ba2)

### Control LLM architecture
![control_llm_architecture](https://github.com/user-attachments/assets/418a4c2b-d94e-4add-8b08-6213a0e6e15e)

### Control LLM Code Base

This repository provides tools to fine-tune LLMs using Control LLM with both pre-training and supervised fine-tuning (SFT). It supports the following features:

- **Flexible Training Options**:
  - Two alternative trainers: native PyTorch trainer and HuggingFace Transformers trainer.

- **Streamlined Configuration**:
  - Default setup for training, dataset, and model-loading configurations.

- **Advanced Data Handling**:
  - Seamless integration with HDFS and any HuggingFace datasets.
  - Support for data preprocessing plugins, feature conversion, caching, and packing.
  - Efficient setup: time to the first training iteration for datasets with 20M+ data points is approximately **2 minutes**.

- **Efficient Distributed Training**:
  - Features include flash attention, model sharding (FSDP/HSFP/DeepSpeed), mixed precision training, gradient accumulation, gradient clipping, parameter-efficient fine-tuning (PEFT), quantized training, multi-node training, and more.
  - Comprehensive debugging and profiling capabilities.

- **Automated Model Evaluation**:
  - Distributed checkpoint saving and conversion.

- **Model Testing and Probing**:
  - Tools for systematic evaluation of model performance and behavior.

- **Benchmarking Support**:
  - Benchmarking capabilities for most open LLM benchmark tasks.

This codebase is designed for efficient and scalable fine-tuning of large language models, enabling researchers and developers to achieve high performance with minimal setup time.

#### Add new dataset from huggingface for training

- check if the dataset already exists in the nfs folder, e.g. /controlllm/huggingface/datasets
- if not, a simple way is to first cache the dataset locally by:
```bash
# load datasets from huggingface
from datasets import load_dataset

dataset = load_dataset("<your datasets>")
# e.g.
dataset = load_dataset("samsum")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k")
dataset = load_dataset("openbmb/UltraFeedback")
dataset = load_dataset("yahma/alpaca-cleaned")
```
- upload the new datasets to nfs
```bash
cp -r ~/.cache/huggingface/datasets/<your datasets>/ /shared/controlllm/huggingface/datasets
# e.g.
cp -r ~/.cache/huggingface/datasets/samsum/ /shared/controlllm/huggingface/datasets

#### Setting up the dev env

- Install git lfs

```bash
check if it is already installed: git lfs install
install lfs if not yet: sudo yum install git-lfs
check if lfs is working: git lfs version
```

- Git clone and pull with lfs
```bash
git clone https://github.com/linkedin/ControlLLM.git
git lfs pull
```

- Pull in submodule(lm-evaluation-harness)
```bash
git submodule update --init --recursive
```

- Create a folder for the script, e.g. /home/jobuser:
```bash
mkdir /home/jobuser/
mkdir /home/jobuser/resources/
```

- Move script to the folder, e.g. /home/jobuser:
```bash
cp -r ControlLLM/src/controlllm /home/jobuser/
cp -r ControlLLM/lm-evaluation-harness /home/jobuser/resources/
cp ControlLLM/wheels/* /home/jobuser/resources/
```

- Check cuda version:

/usr/local/cuda/bin/nvcc --version

** should be 11.8:
... Build cuda_11.8.r11.8/compiler.31833905_0

- manually install the needed python lib:
```bash
cd ControlLLM/src/controlllm
pip install -r requirements.txt

or 
sh /home/jobuser/controlllm/script/100-setup.sh
```

- Test the flash-attn by run python:
```bash
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
```

#### If the dev env is all working, congrats, now run the training pipeline

- trigger the training job
```bash
torchrun --nproc_per_node=8 /home/jobuser/controlllm/main.py
```

#### Iterate and Debug the code
Debug large model that can't be loaded in single GPU(80GB), set this in .vscode/launch.json, together with fsdp or deepspeed in the code, debugger will be launched with model sharding:

```bash
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "/home/jobuser/.local/lib/python3.10/site-packages/torch/distributed/run.py",
            "console": "integratedTerminal",
            "args": [
                "--nproc_per_node=8",
                "${file}"
            ],
            "subProcess": true,
            "internalConsoleOptions": "neverOpen",
            "justMyCode": false
        }
    ]
}
```

- if nvidia-smi shows the process is still running after exit, kill the all python processes by:
```bash
ps aux or ps -al
kill -9 <pid>

# or in batch
pkill -9 -f '/bin/python -u'
pkill -9 -f '/bin/python -c'
pkill -9 -f '/export/apps/python/3.10/bin/python3'
pkill -9 -f '/bin/python'
pkill -9 -f 'ray::run_inference_one_model'
```

#### Train the model

- Detailed Steps:

#### Step 1: go to /home/jobuser/controlllm/scripts/200-run.sh, set up NUM_NODES to 3, uncomment one of following trainer alternative. You only need to specify TRAINER and dataset!

```bash
...
export NUM_NODES=3  # 1 means 1 node, set to 2 for 2 nodes
export TRAINER=native
...

# native torch trainer:
torchrun --nnodes=$NUM_NODES --nproc-per-node=$LOCAL_WORLD_SIZE \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id=1234 --rdzv_backend=c10d \
      /home/jobuser/controlllm/main.py \
      --dataset OpenMathInstruct2Dataset \

# transformer trainer:
torchrun --nnodes=$NUM_NODES --nproc-per-node=$LOCAL_WORLD_SIZE \
      --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" --rdzv_id=1234 --rdzv_backend=c10d \
      /home/jobuser/controlllm/main.py \
      --dataset OpenMathInstruct2Dataset \
```

##### Step 2: set up the right config with best practice:

Note: in order to speed up training, there are 3 different combinations of configurations recommended to set before launching the training with "mldev run"(trainer config is in step 2):

- native trainer with fsdp

go to /home/jobuser/controlllm/config/training.py, set enable_fsdp to True
```bash
    # note: only one of fsdp or deepspeed can be enabled at a time, for transformers trainer, config TrainConfigTransformers->"deepspeed, fsdp, fsdp_config"
    enable_fsdp: bool = True  # enable fsdp for training, for native trainer, fsdp config is in ./configs/fsdp.py
    enable_deepspeed: bool = False  # enable deep speed for training 
```
go to /home/jobuser/controlllm/config/fsdp.py, set hsdp to True and sharding_group_size to 8(8 GPUs per node), replica_group_size to 3(number of nodes)
```bash
    hsdp: bool = True
    # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    sharding_group_size: int = 8 # requires hsdp to be set. Specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int = 3  # requires hsdp to be set. Specifies the replica group size, which is world_size/sharding_group_size.
```

- transformer trainer with fsdp

go to /home/jobuser/controlllm/config/training.py -> TrainConfigCommon, set enable_fsdp to True
```bash
    # note: only one of fsdp or deepspeed can be enabled at a time, for transformers trainer, config TrainConfigTransformers->"deepspeed, fsdp, fsdp_config"
    enable_fsdp: bool = True  # enable fsdp for training, for native trainer, fsdp config is in ./configs/fsdp.py
    enable_deepspeed: bool = False  # enable deep speed for training 
```
go to /home/jobuser/controlllm/config/training.py -> TrainConfigTransformers, set fsdp strategy and config
```bash
    # deepspeed: str = "/home/jobuser/controlllm/configs/z3_++.json"
    fsdp: str = "full_shard auto_wrap"
    fsdp_config: str = "/home/jobuser/controlllm/configs/fsdp.json"
```

- transformer trainer with deepspeed

go to /home/jobuser/controlllm/config/training.py -> TrainConfigCommon, set enable_fsdp to True
```bash
    # note: only one of fsdp or deepspeed can be enabled at a time, for transformers trainer, config TrainConfigTransformers->"deepspeed, fsdp, fsdp_config"
    enable_fsdp: bool = False  # enable fsdp for training, for native trainer, fsdp config is in ./configs/fsdp.py
    enable_deepspeed: bool = True  # enable deep speed for training 
```

go to /home/jobuser/controlllm/config/training.py -> TrainConfigTransformers, set deepspeed config
```bash
    deepspeed: str = "/home/jobuser/controlllm/configs/z3_++.json"
    # fsdp: str = "full_shard auto_wrap"
    # fsdp_config: str = "/home/jobuser/controlllm/configs/fsdp.json"
```

Note: for multi-node training, double check three configs for number of nodes and make sure they are consistent before "mldev run", here is an example for training with 4 nodes.

- go to /home/jobuser/controlllm/configs/fsdp.py, double check "replica_group_size"
```bash
    hsdp: bool = True
    # requires hsdp to be set. This specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    sharding_group_size: int = 8  # requires hsdp to be set. Specifies the sharding group size, number of GPUs that you model can fit into to form a replica of a model.
    replica_group_size: int = 4  # requires hsdp to be set. Specifies the replica group size, which is world_size/sharding_group_size, change this to the actual number of nodes!
```

- go to /home/jobuser/controlllm/scripts/200-run.sh, double check "NUM_NODES"
```bash
    export GPUS_PER_NODE=$(nvidia-smi --list-gpus|wc -l)
    export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
    # Note that to make qgZ work(zero_quantized_gradients): only magic numbers will work, one of [1, 2, 3, 5, 8, 16, 32, 40, 64, 80 ...]
    export NUM_NODES=4  # 1 means 1 node, set to 2 for 2 nodes.
    export WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
```

##### Step 3: launch the training with tmux:

```bash
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# single-node example:
tmux new -s session_name -d "torchrun --nproc-per-node=8 /home/jobuser/controllm/main.py"

# multi-node example:
tmux new -s session_name -d "sh /home/jobuser/controlllm/scripts/100-setup.sh"
tmux new -s session_name -d "sh /home/jobuser/controlllm/scripts/200-run.sh"

to attach:
tmux ls
tmux attach -t session_name

to kill:
tmux kill-session -t session_name
```

- monitor the job via tensorboard:
```bash
# run this in any running pod with interactive dev model, note that the tensorboard writes to /home/jobuser/controlllm/configs/training.py -> output_dir
tensorboard --logdir <output_dir> --port 8081
```

- log and trace:
```bash
# logging:
kubectl cp -c pytorch <pod id>:/var/tmp/log/controlllm . -n kk-flyte-prod

# metrics such as loss curve etc.:
tensorboard --logdir <output_dir>/runs/

# profiler:
pip install torch-tb-profiler
tensorboard --logdir <output_dir>/profiler
```
- Note that for metrics report to tensorboard to wandb to work:
go to /home/jobuser/controlllm/configs/training.py, set enable_tensorboard and use_wandb to True for native trainer and report_to: str = "all" for transformers trainer.

```bash
enable_tensorboard: bool = True  # enable tensorboard for training
use_wandb: bool = False  # enable wandb to log the experiment
```

- Note that for profiler to work:
go to /home/jobuser/controlllm/configs/training.py, set either flop_counter to True or use_profiler to True. flop_counter == True does not work together with triton kernel optimization(/home/jobuser/controlllm/utils/triton_kernels) yet due to one bug in torch, fixed in torch 2.4.1. While waiting for torch 2.4.1 to work with rest of eco-system, flash-attn etc., please disable triton kernel optimization in /home/jobuser/controlllm/main.py by commentting out the code line 21 and 22.

```bash
    enable_tensorboard: bool = True  # enable tensorboard for training
    # ...
    report_to: str = "all"  # trainer knows how to format the output, to print/log training loss during training loop, `"all"`, `"tensorboard"`, `"wandb", `"flyte"`, `"mlflow"`, or `"none"`
    # ...
    flop_counter: bool = False  # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time. For transformers trainer, set to True requires `--include_num_input_tokens_seen` and `logging_steps=1`.
    flop_counter_start: int = 3  # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False  # Enable pytorch profiler, can not be used with flop counter at the same time.
    wait_step, warmup_step, active_step = 1, 2, 4  # The steps to wait, warmup, active, used for pytorch profiler.
    profiler_dir: str = "/home/jobuser/profiler_results"  # will be used if using profiler
```

#### Evaluate the model
- lm-evaluation-harness is used to evaluate the model on openllm leaderboard tasks

```bash
# single GPU:
MODEL_PATH=<model_checkpoint> python python /home/jobuser//controlllm/inference/batch_eval.py

# multiple GPUs:
MODEL_PATH=<model_checkpoint> accelerate launch accelerate launch /home/jobuser/controlllm/inference/batch_eval.py

# in tmux
tmux new -s session_name -d "MODEL_PATH=<model_checkpoint>  accelerate launch /home/jobuser/controlllm/inference/batch_eval.py"
tmux new -s session_name -d "MODEL_PATH=<model_checkpoint> python /home/jobuser/controlllm/inference/batch_eval.py"
tmux attach -t session_name
```

- Note that during auto eval with distributed metrics calculation is enabled during training, and also reported to tensorboard, go to /home/jobuser/controlllm/configs/training.py -> run_validation == True and enable_tensorboard == True, adjust the cadence by eva_steps e.g.

```bash
...
    run_validation: bool = True
    evaluation_strategy: str = "steps"
    # how frequent you want the model to be evaluated or set run_validation to False if you don't want to evaluate
    eval_steps: int = 1000
    # stop by max_eval_step for eval, set to 0 or negative to disable it
    max_eval_step: int = 20
    hf_hub_metrics_cache_dir: str = "/shared/metrics/"  # nfs folder to cache for huggingface metrics, it also caches the code of the metrics calculation which can be customized via remote code
...

# expect this in the training log and all metrics in tensorboard

2024-08-21 05:53:36,836 - root - INFO - global_step=980 eval_ppl=1.0121713876724243 eval_epoch_loss=0.012097976170480251 eval_rouge1=0.9052542481257396 eval_rouge2=0.8054924279938513 eval_rougeL=0.8954994319803518 eval_rougeLsum=0.8957855796276694 eval_bleu=80.51339103358961

# eval rouge1, rouge2, rougeL, rougeLsum, and bleu are computed differently in single node and multi-node for now, single node is more accurate
# so for a trained checkpoint, do a final evaluation by single node with following setting(assuming checkpoint-8500 shows the highest bleu score in tensorboard):
...
    eval_steps: int = 1
    resume_checkpoint_folder: str = "checkpoint-8500"  # "checkpoint-3", change 3 to the global step of the checkpoint you want to load, None to respect resume_from_latest
```

The remote code for computing metrics is in /home/jobuser/controlllm/metrics, copy it over to the configured location of hf_hub_metrics_cache_dir before starting the distributed evaluation or training job.

- Note that it is very likely that training job is stopped in the middle and resume, therefore we need to continously project the eval metrics in the same digram, this is supported as well, expect to see following metrics diagram, taking bleu metrics as an example:

<img width="430" alt="image" src="https://github.com/user-attachments/assets/ac8c36bc-938c-49ef-b011-977caee0b79c">

### Test the model
- go to /home/jobuser/controlllm/inference/chat_completion.py, set vllm_model_path to the model checkpoint directory, for smaller model such as 8b, single GPU is enough, go to ./.vscode/launch.json, set nproc_per_node==1.

```bash
        use_vllm: bool = True,  # Use vllm instead of native transformers
        vllm_model_path: str = "/shared/models/<converted-checkpoint-folder>",  # Model path for vllm, nfs folder
        ori_hf_model_path_or_name: str = "/shared/models/Meta-Llama-3-8B",  # Original HF model path or name, , nfs folder
```

- Note that ori_hf_model_path_or_name is directory where the original model folder downloaded from huggingface. This is needed when we have customized the model architecture, but if there is no change in model architecture, it is fine to keep it the same as vllm_model_path.

- Note that if the checkpoint directory is a saved checkpoint of sharded parameter(e.g. sharded parameter of FSDP), the checkpoint has to be converted to huggingface format. A converter is designed for that, go to /home/jobuser/controlllm/utils/checkpoint_converter.py and execute it by single thread(go to ./.vscode/launch.json and set "--nproc_per_node=1")

```bash
    parser.add_argument("--fsdp_checkpoint_path", type=str, default="", help="Path to FSDP Sharded model checkpoints")
    parser.add_argument("--consolidated_model_path", type=str, default="", help="Path to save the HF converted model checkpoints")
    parser.add_argument("--hf_model_path_or_name", type=str, default="/shared/models/Meta-Llama-3-8B", help="Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)")
```

#### Reference:
- https://github.com/mosaicml/llm-foundry
- https://github.com/togethercomputer/OpenChatKit
- https://github.com/young-geng/EasyLM
- https://github.com/EleutherAI/gpt-neox
- https://github.com/Kipok/NeMo-Skills
- https://github.com/imoneoi/openchat
- https://github.com/facebookresearch/llama-recipes
- https://github.com/kotoba-tech/kotoba-recipes
- https://github.com/jzhang38/TinyLlama
