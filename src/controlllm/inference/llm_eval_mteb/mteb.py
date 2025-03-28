import torch
import logging
from typing import Any
from datetime import datetime

import controlllm.inference.llm_eval_mteb.control_llm_mteb_init  # this applies monkey patching to the MTEB framework to use our custom RetrievalEvaluator
from mteb import MTEB, get_tasks
from mteb.model_meta import ModelMeta

from sentence_transformers.SentenceTransformer import SentenceTransformer

from controlllm.utils import setup_utils


def evaluate_model_mteb(model: SentenceTransformer, model_checkpoint_path: str, tasks: str, per_device_benchmark_batch_size: int, configs, enable_benchmark_debug=False, force_refresh=False):
    """
    Benchmarks a sentence-transformer model using the MTEB framework
    and returns a nested 'results' dictionary.

    Returns:
        dict: {
          "results": {
            "<task_name1>": { "ndcg10": <score> },
            "<task_name2>": { "ndcg10": <score> },
            ...
            "mteb_avg_ndcg10": <average_score or None>
          }
        }
    """
    rank = configs.setup_config.rank
    if rank != 0:
        # Return an empty structure or None for non-master processes
        return {"results": {}}

    # 1 Initialize ModelMeta with the model's metadata
    model.mteb_model_meta = ModelMeta(**initialize_model_meta(configs))

    # 2. Initialize MTEB benchmark with the chosen tasks
    all_tasks = get_tasks(task_types=["Retrieval"])
    if tasks.lower() == "all":
        tasks = all_tasks
    else:
        tasks = tasks.split(",")
        tasks = [t for t in all_tasks if t.__class__.__name__ in tasks]
    evaluation = MTEB(tasks=tasks)

    # 3. Run the evaluation and get the results

    # DDP by num_gpus, tensor_parallel_size=1, without model parallelism
    if enable_benchmark_debug:
        num_gpus = 1  # set it to 1 to debug vLLM with single GPU without ray worker
    else:
        num_gpus = torch.cuda.device_count()
    if num_gpus == 1:  # for multi-process/GPU, control_llm_vllm.py uses ray so leave it to each worker to do init_distributed_mode. for single process, set up GPU here
        setup_utils.init_distributed_mode(configs.setup_config)

    # Customized MTEB to support DDP with data_parallel_size of all available GPUs
    encode_kwargs: dict[str, Any] = {"data_parallel_size": num_gpus, "model_checkpoint_path": model_checkpoint_path, "batch_size": per_device_benchmark_batch_size}
    mteb_results = evaluation.run(model, output_folder=model_checkpoint_path, overwrite_results=force_refresh, encode_kwargs=encode_kwargs)

    # 4. Extract each task's main score (nDCG@10 for retrieval tasks)
    retrieval_scores = {
        task_result.task_name: task_result.get_score()
        for task_result in mteb_results
    }
    logging.info(f"MTEB benchmark results (raw): {retrieval_scores}")

    # 4. Compute the average retrieval score
    if retrieval_scores:
        avg_retrieval_score = sum(retrieval_scores.values()) / len(retrieval_scores)
    else:
        avg_retrieval_score = None

    logging.info(f"Average Retrieval Score (nDCG@10): {avg_retrieval_score}")

    # 6. Build the 'results' dict
    #    - Each task has a nested dict with a key like "ndcg10".
    #    - We also include an overall "mteb_avg_ndcg10" in the same dict.
    # Example structure:
    # {
    #   "results": {
    #     "msmarco":  {"ndcg10": 0.80},
    #     "scidocs":  {"ndcg10": 0.60},
    #     ...
    #     "mteb_avg_ndcg10": 0.70
    #   }
    # }
    results_dict = {
        "results": {
            **{task_name: {"ndcg10": score} for task_name, score in retrieval_scores.items()},
            "mteb_avg_ndcg10": avg_retrieval_score
        }
    }

    return results_dict


def initialize_model_meta(configs):
    model_checkpoint_path = configs.model_loading_config.pretrained_model_name_or_path  # e.g. /shared/model/hf-checkpoint-sft-padding-semantic-margin-stack-cosent/checkpoint-34000
    global_step = int(model_checkpoint_path.rstrip('/').rsplit('checkpoint-', 1)[1]) if 'checkpoint-' in model_checkpoint_path else 0
    model_name = model_checkpoint_path.split('/')[-2]

    # Prepare the data, using None for missing fields
    model_data = {
        "name": model_name,
        "revision": str(global_step),
        "release_date": datetime.now().strftime("%Y-%m-%d"),
        "languages": ["eng"],
        "framework": ["PyTorch"],
        "loader": None,
        "n_parameters": None,
        "memory_usage_mb": None,
        "max_tokens": None,
        "embed_dim": None,
        "license": None,
        "open_weights": None,
        "public_training_code": None,
        "public_training_data": None,
        "similarity_fn_name": None,
        "reference": None,
        "use_instructions": None,
        "training_datasets": None,
        "adapted_from": None,
        "superseded_by": None,
        "modalities": ["text"]
    }

    return model_data
