import os
import uuid
from pathlib import Path

from evaluate import EvaluationModule, load


def initialize_metrics_modules(module="code_eval", hf_metrics_cache=os.environ["HF_METRICS_CACHE"]) -> EvaluationModule:
    """
    Initialize the metrics modules for coding, disable distributed computing as pass@k is parallelizable and set the timeout to 10 seconds.
    """
    assert "HF_METRICS_CACHE" in os.environ, "Please set HF_METRICS_CACHE to the path of the cache directory of code_eval module."
    pass_at_k: EvaluationModule = load(path=str(Path(hf_metrics_cache) / module), keep_in_memory=True, trust_remote_code=True, timeout=10)

    return pass_at_k
