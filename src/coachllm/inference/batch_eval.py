import os
import re
import json
import logging
import subprocess
from tqdm import tqdm
from typing import List
from pathlib import Path

# don't try to download datasets from internet, it takes too long to fail and fallback to cache
if "HF_DATASETS_OFFLINE" not in os.environ:
    os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["NCCL_TIMEOUT"] = "36000"
# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"  # make gemma model work with vLLM, refer to: https://github.com/vllm-project/vllm/issues/6220
os.environ["WANDB_DISABLED"] = "true"  # disable wandb, only enable it when GPU cluster of training is set up with https access
os.environ['RAY_DEDUP_LOGS'] = '0'  # disable deduplication of logs in Ray, so that we can see the full logs of all workers
os.environ['NUMEXPR_MAX_THREADS'] = '64'  # System dependent, A100 has 256 virtual cores available, so safe to set it to 64
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # avoid memory fragmentation following https://github.com/pytorch/pytorch/issues/130330
os.environ['MODULE'] = 'controlllm_eval'
if "MODEL_PATH" not in os.environ:
    os.environ['MODEL_PATH'] = ""  # set the model path here, otherwise run with MODEL_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py

import fire
# make sure llm_eval_harness.eval is imported before datasets lib gets imported in other import because os.environ["HF_DATASETS_CACHE"] needs to be set before importing the datasets library
from controlllm.inference.llm_eval_harness.eval import evaluate_model, parse_eval_args

# apply triton kernel customizations, it is particularly useful for llama 3 models as vocab size is 13k+
from controlllm.utils.triton_kernels import apply_model_customizations
apply_model_customizations()

import torch
from dataclasses import asdict
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from controlllm.utils import setup_utils
from controlllm.utils.config_utils import Configs
from controlllm.utils.model_expander import ModelExpander, ModelComparer
from controlllm.utils.loading_utils import ModelLoader
from controlllm.utils.dataset_utils import DataLoaderWrapper
from controlllm.utils.checkpoint_converter import load_model_from_config
from controlllm.utils.custom_llama_recipes.model_checkpointing import load_sharded_model_single_gpu
from controlllm.utils.custom_llama_recipes.eval_utils import evaluate
from controlllm.inference.chat_completion import InferenceEngine


def main(**kwargs):
    # Handle the MODEL_PATH in os.environ to support MODEL_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py
    if "MODEL_PATH" in os.environ:
        kwargs = handle_model_path_in_os_env(**kwargs)
        if "resume_checkpoint_folder" in kwargs and isinstance(kwargs["resume_checkpoint_folder"], list):
            resume_checkpoint_folders = kwargs.get('resume_checkpoint_folder', '')
            logging.info(f"MODEL_PATH in os.environ detected, output_dir: {kwargs.get('output_dir', '')}, resume_checkpoint_folder: {resume_checkpoint_folders}. Running them one by one")
            for resume_checkpoint_folder in tqdm(resume_checkpoint_folders):
                try:
                    kwargs["resume_checkpoint_folder"] = resume_checkpoint_folder
                    logging.info(f"Running batch evaluation with output_dir: {kwargs.get('output_dir', '')}, resume_checkpoint_folder: {kwargs.get('resume_checkpoint_folder', '')}")
                    evaluation_engine = EvaluationEngine(**kwargs)
                    if evaluation_engine.run_eval:
                        evaluation_engine.run_evaluation()
                    if evaluation_engine.run_leaderboard:
                        evaluation_engine.run_benchmark()
                    torch.cuda.empty_cache()  # This can help release unoccupied memory back to the GPU
                except Exception as e:
                    logging.exception(f"Error: {e}")
                    logging.error(f"Error running batch evaluation with output_dir: {kwargs.get('output_dir', '')}, resume_checkpoint_folder: {kwargs.get('resume_checkpoint_folder', '')} - Continue to next checkpoint")

            return

    evaluation_engine = EvaluationEngine(**kwargs)

    if evaluation_engine.run_eval:
        evaluation_engine.run_evaluation()

    if evaluation_engine.run_leaderboard:
        evaluation_engine.run_benchmark()


class EvaluationEngine(InferenceEngine):
    """
    evaluation engine for the chat completion
    """

    def __init__(
        self,
        use_cache: bool = True,  # Whether the model should use the past last key/values attentions to speed up decoding.
        torch_dtype: str = "bf16",  # The data type to use to load the model, fp16, bf16 or fp32
        enable_fsdp: bool = False,  # Whether to enable Fully Sharded Data Parallelism for model loading
        output_dir: str = "",  # which checkpoint to evaluate on, set this to the output directory of the model and specify resume_checkpoint_folder if needed
        resume_checkpoint_folder: str = None,  # "checkpoint-3", change 3 to the global step of the checkpoint you want to load, None to respect resume_from_latest
        model_checkpoint_path: str = None,  # The path to the model checkpoint to load, note that either use "output_dir + resume_checkpoint_folder" or model_checkpoint_path with full model path which has to be converted to huggingface format by src/controlllm/utils/checkpoint_converter.py
        per_device_eval_batch_size: int = 1,  # The batch size for evaluation
        max_eval_step: int = -1,  # stop by max_eval_step for eval, set to 0 or negative to disable it
        eval_datasets: List[str] = None,  # The datasets to evaluate on, leave it to None to eval on trained_by datasets set in the model config, must be list of class names from .controlllm/configs/datasets.py, e.g. ['OpenMathInstruct2Dataset', 'OpenCoderSFTStage2']
        batching_strategy: str = "padding",  # padding or packing
        run_eval: bool = False,  # Whether to run evaluation on eval_datasets defined above, all above parameters are used for evaluation if this is True
        run_model_debug: bool = False,  # Whether to run model debug, only used when run_eval is True. Please ensure the data is small in size to avoid overwhelming TensorBoard. e.g. by setting max_eval_step to small number like 20 or make probe_data small when use_probe_data is True
        use_probe_data: bool = False,  # Whether to use probe data for model debug, only used when run_model_debug is True, probe_data: refer to it in controlllm/utils/custom_llama_recipes/eval_utils.py.
        output_attentions: bool = False,  # Whether to output attentions, used when run_model_debug is True to plot attentions
        run_leaderboard: bool = True,  # Whether to run benchmark on tasks defined below, all below parameters are used for benchmark if this is True
        use_vllm: bool = True,  # Whether to use vllm to run benchmark, if False, use the default "hf"(huggingface transformers lib) to run benchmark. If True, run benchmark with single thread "python ~/controlllm/inference/batch_eval.py". If False, run it with multi-threading "accelerate launch ~/controlllm/inference/batch_eval.py"
        apply_chat_template: bool = False,  # Whether to apply chat template to the chat history. Note that only instruct models require this. Some tasks(everything starting with meta) are already using chat template in the dataset, so set it to False for those tasks.
        fewshot_as_multiturn: bool = False,  # Whether to treat fewshot as multiturn, only used when apply_chat_template is True
        merge_layers: bool = False,  # Whether to merge the layers of the model, only used when use_vllm is True and model checkpoint is sharded
        # here are the benchmark tasks recommended by Control LLM, *_pretrain for pretrain, *_instruct for instruct, *_multishot for multishot:
        # tasks="arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,mathqa,mmlu,mmlu_pro_5shot,ceval-valid,cmmlu",  # This is the benchmark tasks from lm-evaluation-harness
        # tasks="code_pretrain,code_instruct,math_pretrain,math_instruct,zh_pretrain,zh_instruct,meta_pretrain,meta_instruct,leaderboard,original_capability_instruct,original_capability_pretrain",  # This is the 0shot benchmark tasks
        # tasks="code_pretrain_multishot,code_instruct_multishot,math_pretrain_mutlishot,math_instruct_multishot,zh_pretrain_multishot",  # This is the mutlishot benchmark tasks
        # tasks: str = "original_capability_instruct,math_instruct,code_instruct",  # this is to reproduce Control LLM reported result for math and coding
        # tasks: str = "meta_pretrain,zh_pretrain_multishot",  # this is to reproduce Control LLM reported result for Chinese
        tasks: str = "original_capability_instruct,math_instruct,code_instruct,zh_instruct",  # The tasks to run benchmark on, set to None to use default in llm_eval_harness.eval->eval.py
        force_refresh: bool = False,  # Whether to force refresh benchmark results. If False, skip run_leaderboard when file_name = f"benchmark_results_{args.tasks.replace(',', '_')}.json" already exists in model_checkpoint_path
        per_device_benchmark_batch_size: int = 8,  # The batch size for benchmark of open llm leaderboard, used when run_leaderboard is True. For "mmlu,ceval-valid,cmmlu", set it to 1
        max_model_len: int = 8192,  # The maximum model length for vLLM, only used when use_vllm is True. For "code_pretrain,code_instruct", use 10192. For rest, use 8192. If vLLM model init runs OOM, consider reducing it
        cpu_offload_gb: int = 0,  # The amount of memory to offload weights to CPU in GB, only used when use_vllm is True. When OOM, consider increasing it to 10.
        gpu_memory_utilization: float = 0.8,  # The amount of GPU memory to utilize, only used when use_vllm is True. For A100, set it to 0.8. For "mmlu,ceval-valid,cmmlu", set it to 0.6.
        enable_prefix_caching: bool = True,  # Whether to enable prefix caching, only used when use_vllm is True
        enable_benchmark_debug: bool = False,  # Whether to enable debug for benchmark with vLLM, only used when use_vllm is True. Only set to True for debugging purpose, run benchmark with False. Note that ray is used to run vLLM, to debug the code, need to set to only 1 GPU to disable ray worker.
        compare_weight: bool = False,  # Whether to compare the weights of the model with the original model to make sure the conversion is correct
        **kwargs  # Accepts any additional keyword arguments
    ):
        # Assign input parameters to class attributes
        self.use_cache = use_cache
        self.enable_fsdp = enable_fsdp
        self.run_eval = run_eval
        self.run_model_debug = run_model_debug
        self.run_leaderboard = run_leaderboard
        self.use_vllm = use_vllm
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn,
        self.merge_layers = merge_layers
        self.tasks = tasks
        self.force_refresh = force_refresh
        self.per_device_benchmark_batch_size = per_device_benchmark_batch_size
        self.max_model_len = max_model_len
        self.cpu_offload_gb = cpu_offload_gb
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_benchmark_debug = enable_benchmark_debug
        self.compare_weight = compare_weight
        self.use_probe_data = use_probe_data
        self.output_attentions = output_attentions
        self.kwargs = kwargs

        # Load the default configuration parameters and update them with the command line arguments
        self.configs = Configs(**kwargs)

        # How to load the model
        self.configs.model_loading_config.torch_dtype = torch_dtype  # hardcode model loading to bf16 to reduce memory as it becomes the standard for open source model
        self.configs.model_loading_config.use_cache = self.use_cache
        self.configs.model_loading_config.output_attentions = self.output_attentions
        if self.output_attentions:  # output attentions requires eager mode
            self.configs.model_loading_config.attn_implementation = "eager"
        # How many expansion layers were added to the model
        if model_checkpoint_path:
            self.model_checkpoint_path = model_checkpoint_path
        else:
            if not output_dir:
                raise ValueError("either model_checkpoint_path or output_dir is required to run evaluation")
            self.model_checkpoint_path = get_last_checkpoint(output_dir) if resume_checkpoint_folder is None else str(Path(output_dir) / resume_checkpoint_folder)
            if self.model_checkpoint_path is None:
                raise ValueError(f"No checkpoint found in {output_dir}, it should have different checkpoint folders like 'checkpoint-1', 'checkpoint-2', ...")
        self.global_step = int(self.model_checkpoint_path.rstrip('/').rsplit('checkpoint-', 1)[1]) if 'checkpoint-' in self.model_checkpoint_path else 0  # assuming the checkpoint path is always ending with checkpoint-<global_step>
        self.configs.model_loading_config = ModelExpander.restore_expansion_configs(self.model_checkpoint_path, self.configs.model_loading_config)
        self.configs.model_loading_config.__post_init__()
        # model_loading_config.__post_init__() converts the torch_dtype to torch.dtype, so we need to update it here
        self.torch_dtype = self.configs.model_loading_config.torch_dtype

        # How to evaluate the model
        self.configs.train_config.run_validation = True
        self.configs.train_config.enable_fsdp = enable_fsdp
        self.configs.train_config.per_device_eval_batch_size = per_device_eval_batch_size
        self.configs.train_config.max_eval_step = max_eval_step
        self.configs.train_config.batching_strategy = batching_strategy

        # Sharded weights does not work with kv cache
        if self.configs.train_config.enable_fsdp or self.configs.train_config.enable_deepspeed:
            self.configs.model_loading_config.use_cache = False

        # Which datasets to evaluate on
        if self.run_eval:
            self.trained_by = ModelLoader.get_trained_by_datasets(self.model_checkpoint_path)
            if eval_datasets is None:
                if self.trained_by is None:
                    raise ValueError(f"Could not find the 'trained_by' in the config of checkpoint: {self.model_checkpoint_path}, double check if the folder exists or checkpoint exists in format of checkpoint-1, checkpoint-2, ...")
                else:
                    eval_datasets = self.trained_by
            self.configs.train_config.dataset = eval_datasets
            self.configs.generate_dataset_cfg(**kwargs)
            for dataset in eval_datasets:
                matching_configs = [config for config in self.configs.dataset_configs if config.dataset == dataset]
                for config in matching_configs:
                    config.run_validation = True

        # Which base model it was trained from to resgister the expanded model classes
        self.trained_from = ModelLoader.get_trained_from_model_name_or_path(self.model_checkpoint_path)
        if self.trained_from is None:
            logging.warning(f"Could not find the 'trained_from' in the config of checkpoint: {self.model_checkpoint_path}, double check if the folder exists or checkpoint exists in format of checkpoint-1, checkpoint-2, ...")

        if self.run_eval and self.use_vllm:
            raise ValueError("vLLM is supported only for running benchmark, please set run_eval=False or use_vllm=False")

        logging.info(f"Registering the expanded model classes with new model architecture from {self.model_checkpoint_path}")
        ModelExpander.register_expansion_classes(self.model_checkpoint_path, self.use_vllm)

        # Which model to load for evaluation
        if model_checkpoint_path:
            self.configs.train_config.output_dir = self.model_checkpoint_path
            self.configs.train_config.resume_checkpoint_folder = None
        else:
            if not output_dir:
                raise ValueError("output_dir is required to run evaluation")
            self.configs.train_config.output_dir = output_dir
            self.configs.train_config.resume_checkpoint_folder = resume_checkpoint_folder
        self.configs.model_loading_config.pretrained_model_name_or_path = self.model_checkpoint_path  # self.model_checkpoint_path is not yet converted to huggingface format, we will have special handling in run_evaluation()
        self.configs.tokenizer_loading_config.pretrained_model_name_or_path = self.model_checkpoint_path
        self.configs.train_config.logging_dir = self.configs.train_config.profiler_dir = None  # make sure eval logging to output_dir
        self.configs.train_config.__post_init__()

        # How to report the metrics
        self.wandb_run = setup_utils.setup_wandb(self.configs.wandb_config, self.configs.train_config, self.configs.fsdp_config, self.configs.setup_config.rank)
        self.tb_writer = setup_utils.setup_tensorboard(self.configs.train_config, self.configs.setup_config.rank)

        # Set it up to enable runing and debugging in cpu, gpu or distributed mode
        if not self.use_vllm:  # vllm uses Ray for multiprocessing by DDP, so don't set up distributed mode for vllm
            setup_utils.init_distributed_mode(self.configs.setup_config)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"  # vllm uses Ray for multiprocessing, so set TOKENIZERS_PARALLELISM=false to avoid potential issues related to threading and process forking

    def run_evaluation(self):
        """
        Run the evaluation loop.
        """
        if (Path(self.model_checkpoint_path) / "sharded_checkpoint.txt").exists() or self.enable_fsdp:  # TODO: handle other types of checkpoint in checkpoint_handler->load_checkpoint
            if self.enable_fsdp:  # reuse the model loading logic for training for now
                # Load the huggingface formated model from trained_from and then update the model with the sharded weights in output_dir and resume_checkpoint_folder
                self.configs.model_loading_config.pretrained_model_name_or_path = self.trained_from
                model_loader = ModelLoader(self.configs)
                model = model_loader.model
                tokenizer = model_loader.tokenizer
            else:
                logging.info(f"Loading model from {self.model_checkpoint_path}")
                model = load_model_from_config(self.model_checkpoint_path)
                load_sharded_model_single_gpu(model, self.model_checkpoint_path, False)
                model = model.to(device=self.configs.setup_config.device, dtype=self.torch_dtype)
                model.config.use_cache, model.config.output_attentions = self.use_cache, self.output_attentions
                if self.output_attentions:  # output attentions requires eager mode
                    model.config.attn_implementation = "eager"
                logging.info(f"Loading tokenizer from {self.configs.model_loading_config.pretrained_model_name_or_path}")
                tokenizer = AutoTokenizer.from_pretrained(**asdict(self.configs.tokenizer_loading_config))
        else:
            logging.info(f"Loading model from {self.configs.model_loading_config.pretrained_model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(**asdict(self.configs.model_loading_config))
            model = model.to(device=self.configs.setup_config.device, dtype=self.torch_dtype)
            model.config.use_cache, model.config.output_attentions = self.use_cache, self.output_attentions
            if self.output_attentions:  # output attentions requires eager mode
                model.config.attn_implementation = "eager"
            logging.info(f"Loading tokenizer from {self.configs.model_loading_config.pretrained_model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(**asdict(self.configs.tokenizer_loading_config))

        if self.compare_weight and self.configs.setup_config.rank == 0:
            # Compare the weights of the model with the original model to make sure the conversion is correct
            ModelComparer(model, self.pretrained_model_name_or_path).compare_model_weights()

        # Load and preprocess the datasets
        self.configs.train_config.enable_fsdp = True  # special handling to enable distributed data sampler and evalution regardless if model is loaded with fsdp or not
        if not hasattr(self, 'dataloader'):  # ensure that an attribute is initialized only once
            self.configs.tokenizer_loading_config.pretrained_model_name_or_path = self.trained_from  # make sure cached dataset is loaded with the same tokenizer as the trained_from model
            self.data_loader = DataLoaderWrapper(self.configs, tokenizer)

        eval_ppl, eval_loss, eval_bleu, eval_rougeLsum, eval_step_loss, eval_step_perplexity = evaluate(
            model=model,
            train_config=self.configs.train_config,
            eval_dataloader=self.data_loader.eval_dataloader,
            local_rank=self.configs.setup_config.local_rank,
            rank=self.configs.setup_config.rank,
            tokenizer=tokenizer,
            wandb_run=self.wandb_run,
            tb_writer=self.tb_writer,
            global_step=self.global_step,
            enable_model_debug=self.run_model_debug,
            use_probe_data=self.use_probe_data
        )
        del model
        result = {
            "eval_ppl": eval_ppl,
            "eval_loss": eval_loss,
            "eval_bleu": eval_bleu,
            "eval_rougeLsum": eval_rougeLsum,
            "eval_step_loss": eval_step_loss,
            "eval_step_perplexity": eval_step_perplexity,
        }

        if self.configs.setup_config.rank == 0 and not self.run_model_debug:
            output_path = self.model_checkpoint_path
            if Path(output_path).exists():
                with open(Path(output_path) / "evaluation_results.json", "w") as f:
                    json.dump(result, f, indent=4)
                logging.info(f"Saved evaluation results to {output_path}/evaluation_results.json")
            else:
                logging.warning(f"Checkpoint directory {output_path} does not exist, skipping saving evaluation results")

            logging.info(f"Evaluation finished - eval results: {result}")

    def run_benchmark(self):
        """
        Run the benchmark loop.
        """
        if (Path(self.model_checkpoint_path) / "sharded_checkpoint.txt").exists() and self.use_vllm:
            logging.info(f"vLLM does not support sharded weights in {self.model_checkpoint_path}, convert the model to huggingface format first...")
            consolidated_model_path = self.convert_model_to_huggingface_format()
            logging.info(f"Updating model_checkpoint_path from {self.model_checkpoint_path} to {consolidated_model_path} for running benchmark in vLLM")
            self.model_checkpoint_path = str(consolidated_model_path)

        args = parse_eval_args()
        # Extract just 'bfloat16' by splitting the string
        args.model_args = f"pretrained={self.model_checkpoint_path},dtype={str(self.torch_dtype).split('.')[-1]},trust_remote_code=True"  # e.g. Extract just 'bfloat16' from torch.bfloat16
        args.output_path = self.model_checkpoint_path
        args.batch_size = self.per_device_benchmark_batch_size  # allow different batch size for benchmark from eval to speed up
        args.fewshot_as_multiturn = self.fewshot_as_multiturn
        args.apply_chat_template = self.apply_chat_template  # apply chat template to the chat history, note that only instruct models require this.
        if self.tasks:
            args.tasks = self.tasks
        logger = logging.getLogger()

        benchmark_result_filepath = Path(args.output_path) / f"benchmark_results_{args.tasks.replace(',', '_')}.json"
        if benchmark_result_filepath.exists() and not self.force_refresh:
            logger.warning(f"Benchmark result for {args.tasks} already exists at {benchmark_result_filepath}. Skip the benchmark run, set force_refresh to True to force the run_benchmark ...")
            return

        if (Path(self.model_checkpoint_path) / "sharded_checkpoint.txt").exists():
            args.model_args += f",{'' if args.apply_chat_template else 'add_bos_token=True'}"  # add_bos_token=True is required for models trained by this codebase.

            if self.enable_fsdp:  # reuse the model loading logic for training for now
                # Load the huggingface formated model from trained_from and then update the model with the sharded weights in output_dir and resume_checkpoint_folder
                self.configs.model_loading_config.pretrained_model_name_or_path = self.trained_from
                model_loader = ModelLoader(self.configs)
                model = model_loader.model
            else:
                logging.info(f"Loading model from {self.model_checkpoint_path}")
                model = load_model_from_config(self.model_checkpoint_path)
                load_sharded_model_single_gpu(model, self.model_checkpoint_path, False)
                model = model.to(device=self.configs.setup_config.device, dtype=self.torch_dtype)
                model.config.use_cache = self.use_cache

            results = evaluate_model(
                args=args,
                logger=logger,
                loaded_model=model,
                rank=self.configs.setup_config.rank,
            )
        else:
            if self.use_vllm:
                args.model = "vllm"
                # DDP by num_gpus, tensor_parallel_size=1, without model parallelism
                if self.enable_benchmark_debug:
                    num_gpus = 1  # set it to 1 to debug vLLM with single GPU without ray worker
                else:
                    num_gpus = torch.cuda.device_count()
                if num_gpus == 1:  # for multi-process/GPU, llama_plus_vllm.py uses ray so leave it to each worker to do init_distributed_mode. for single process, set up GPU here
                    setup_utils.init_distributed_mode(self.configs.setup_config)
                # note: trained_from is required for vLLM to register the expanded model classes, for none expanded model class, it is not required but still okay to provide. TODO: check if it is required, remove register_model={llama_plus_model} if not required
                # note: As for `add_bos_token=True`, since our prompts in the evals dataset has already included all the special tokens required by instruct model, such as `<|start_header_id|>user<|end_header_id|>`, we will not use `--apply_chat_template` argument for instruct models anymore. However, we need to use `add_bos_token=True` flag to add the BOS_token back during VLLM inference, as the BOS_token is removed by default in [this PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1465).
                # TODO: make this as input parameter instead of here
                args.model_args += f",tensor_parallel_size=1,gpu_memory_utilization={self.gpu_memory_utilization},data_parallel_size={num_gpus},max_model_len={self.max_model_len},cpu_offload_gb={self.cpu_offload_gb},enable_prefix_caching={self.enable_prefix_caching},{'' if self.apply_chat_template else 'add_bos_token=True,'}seed=42,{('register_model=' + self.model_checkpoint_path) if self.trained_from and num_gpus > 1 else ''}"
            else:
                args.model_args += f",{'' if args.apply_chat_template else 'add_bos_token=True'}"  # add_bos_token=True is required for models trained by this codebase.

            results = evaluate_model(
                args=args,
                logger=logger,
                rank=self.configs.setup_config.rank,
            )

        if self.configs.setup_config.rank == 0:
            results = results["results"]
            logging.info(f"Benchmark results of {self.global_step}: {results}")
            # select some keys to show in the leaderboard from results, by e.g. args.tasks = "arc_challenge,hellaswag,truthfulqa_mc2,winogrande,gsm8k,mmlu"
            leaderboard_results = {key: results[key] for key in args.tasks.split(",") if key in results}
            logging.info(f"Logging benchmark results to wandb and tensorboard: {leaderboard_results}")
            for leaderboard_result in leaderboard_results:
                # take the first key in the dict
                if isinstance(leaderboard_results[leaderboard_result], dict) and len(leaderboard_results[leaderboard_result]) >= 1:
                    first_key = list(leaderboard_results[leaderboard_result].keys())[0]  # take the first key in the dict, e.g. {'arc_challenge': {'accuracy': 0.5}} -> accuracy
                    if first_key == ' ':  # for group that does not have aggregate such as leaderboard, it is {'leaderboard': {' ': ' '}
                        leaderboard_results[leaderboard_result] = -1  # -1 means no value
                    else:
                        leaderboard_results[leaderboard_result] = leaderboard_results[leaderboard_result][first_key]
                else:
                    leaderboard_results[leaderboard_result] = -1  # -1 means no value
            if self.wandb_run:
                self.wandb_run.log({**leaderboard_result, 'global_step': self.global_step}, commit=False)
            if self.tb_writer:
                for key, value in leaderboard_results.items():
                    self.tb_writer.add_scalar(f"Benchmark-{key}/GlobalStep", value, self.global_step)

    def convert_model_to_huggingface_format(self):
        """
        Convert the model checkpoint to huggingface format by checkpoint_converter.py
        """
        logging.info(f"Converting model checkpoint to huggingface format: {self.model_checkpoint_path}...")

        # check if self.model_checkpoint_path is ending with "checkpoint-<global_step>" <global_step> is a number
        if not re.match(r".*checkpoint-\d+$", self.model_checkpoint_path):
            raise ValueError(f"model_checkpoint_path: {self.model_checkpoint_path} is not in the format of 'checkpoint-<global_step>'")
        output_dir = self.model_checkpoint_path.rsplit("checkpoint-", 1)[0]
        output_folder = f"checkpoint-{self.model_checkpoint_path.rsplit('checkpoint-', 1)[1]}"
        consolidated_model_path = Path(output_dir) / ("hf-merged" if self.merge_layers else "hf") / output_folder

        if not consolidated_model_path.exists():
            process = subprocess.run(
                [
                    arg for arg in [
                        'python',
                        'controlllm/utils/checkpoint_converter.py',
                        '--fsdp_checkpoint_path', self.model_checkpoint_path,
                        '--consolidated_model_path', str(consolidated_model_path),
                        '--merge_layers' if self.merge_layers else None
                    ] if arg
                ],
                check=True
            )
            process.check_returncode()  # check if the process was successful

            logging.info(f"Successfully converted model checkpoint to huggingface format in: {consolidated_model_path}")
        else:
            logging.info(f"Model checkpoint already exists in huggingface format in: {consolidated_model_path}")

        return consolidated_model_path


def handle_model_path_in_os_env(**kwargs):
    """
    Handle the MODEL_PATH in os.environ to support MODEL_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py
    """
    # split os.environ["MODEL_PATH"] into the output_dir and resume checkpoint folder
    model_checkpoint_path = os.environ.get("MODEL_PATH", "")
    if model_checkpoint_path:
        # if model_checkpoint_path is ending with "checkpoint-<global_step>" <global_step> is a number
        if re.match(r".*checkpoint-\d+$", model_checkpoint_path):  # don't put / in the end of the path for single checkpoint evaluation
            output_dir = model_checkpoint_path.rsplit("checkpoint-", 1)[0]
            resume_checkpoint_folder = f"checkpoint-{model_checkpoint_path.rsplit('checkpoint-', 1)[1]}"
            if Path(output_dir) / resume_checkpoint_folder != Path(model_checkpoint_path):  # not in format of <output_dir>/checkpoint-<global_step>
                kwargs["model_checkpoint_path"] = model_checkpoint_path
                logging.info(f"MODEL_PATH in os.environ detected, model_checkpoint_path: {kwargs.get('model_checkpoint_path', '')}")
            else:
                kwargs["output_dir"] = output_dir
                kwargs["resume_checkpoint_folder"] = resume_checkpoint_folder
                logging.info(f"MODEL_PATH in os.environ detected, output_dir: {kwargs.get('output_dir', '')}, resume_checkpoint_folder: {kwargs.get('resume_checkpoint_folder', '')}")
        # else if model_checkpoint_path is a folder with model weights, use it as model_checkpoint_path
        elif Path(model_checkpoint_path).is_dir() and (Path(model_checkpoint_path) / "config.json").exists():
            kwargs["model_checkpoint_path"] = model_checkpoint_path
            logging.info(f"MODEL_PATH in os.environ detected, model_checkpoint_path: {kwargs.get('model_checkpoint_path', '')}. It has config.json file, so assuming it is a model checkpoint folder with model weights")
        # if model_checkpoint_path is a folder without model weights, try to get all all the checkpoints in the folder
        else:
            all_checkpoints = get_all_checkpoints(model_checkpoint_path)
            kwargs["output_dir"] = model_checkpoint_path
            kwargs["resume_checkpoint_folder"] = all_checkpoints  # run all the checkpoints in the folder
    else:
        logging.warning("MODEL_PATH not found in os.environ, please run with MODEL_PATH=<model checkpoint path> accelerate launch ./src/controlllm/inference/batch_eval.py")

    return kwargs


def get_all_checkpoints(folder) -> List[str]:
    """
    Get all the checkpoints in the folder.
    """
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")

    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)

    return checkpoints


if __name__ == "__main__":
    fire.Fire(main)
