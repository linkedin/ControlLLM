import os
import re
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from more_itertools import distribute

from lm_eval.api.registry import register_model
from lm_eval.models.utils import undistribute
from lm_eval.models.vllm_causallms import VLLM

try:
    import ray
    from vllm import LLM, SamplingParams
except ModuleNotFoundError:
    pass

from controlllm.utils.model_expander import ModelExpander

if TYPE_CHECKING:
    pass


@register_model("control_llm_vllm")
class ControlLLMWrapper(VLLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.data_parallel_size <= 1:
            pass
        else:
            # to disable worker_use_ray in vLLM: Let Ray handle the workers externally, and ensure that vLLM does not try to manage Ray internally
            self.model_args["worker_use_ray"] = False

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        if len(chat_history) == 0:
            return ""

        input_prompt = self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )
        default_system_prompt_to_be_removed_pattern = r"<\|start_header_id\|>system<\|end_header_id\|>\n\nCutting Knowledge Date: .*\nToday Date: \d{2} \w{3} \d{4}\n\n<\|eot_id\|>"
        # Remove the matching system prompt from input_prompt
        input_prompt = re.sub(default_system_prompt_to_be_removed_pattern, "", input_prompt)

        return input_prompt

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        if generate:
            kwargs = self.modify_gen_kwargs(kwargs)
            sampling_params = SamplingParams(max_tokens=max_tokens, stop=stop, **kwargs)
        else:
            sampling_params = SamplingParams(
                temperature=0, prompt_logprobs=1, max_tokens=1, detokenize=False
            )
        if self.data_parallel_size > 1:
            # vLLM hangs if tensor_parallel > 1 and resources are set in ray.remote
            # also seems to only work with decorator and not with ray.remote() fn
            # see https://github.com/vllm-project/vllm/issues/973
            # note: this has changed on 0.3.3, and it only works now if num_gpus are set.
            # but then tensor_parallel breaks

            # Customize it to one GPU per worker, good enough for 10B model as it can be loaded in single GPU, and it is faster than multi-GPU, change this to your needs if needed
            import subprocess
            # First, shutdown Ray in the current script if it's running
            if ray.is_initialized():
                ray.shutdown()
            # Then, use subprocess to call 'ray stop' and terminate all Ray processes
            subprocess.call(['ray', 'stop'])
            ray.init(num_gpus=self.data_parallel_size, local_mode=False)
            print(ray.available_resources())

            @ray.remote(num_gpus=1, num_cpus=8)
            def run_inference_one_model(
                model_args: dict, sampling_params, requests: List[List[int]]
            ):
                from ray import get_gpu_ids
                gpu_ids = get_gpu_ids()
                pid = os.getpid()
                print(f"Worker PID: {pid}, assigned GPU(s): {gpu_ids}")
                # DDP without model parallelism
                if gpu_ids:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_ids[0]))
                # register expansion classes per ray worker
                if "register_model" in model_args:
                    register_model = model_args.pop("register_model")
                    ModelExpander.register_expansion_classes(register_model, use_vllm=True)
                llm = LLM(**model_args)
                print(f"Worker PID: {pid}, assigned GPU(s): {gpu_ids}, model loaded successfully.")
                return llm.generate(
                    prompt_token_ids=requests, sampling_params=sampling_params
                )

            # dispatch requests to all self.data_parallel_size workers, in interleaved fashion
            if not requests:
                logging.warning("No inference requests available to process.")
                return []
            # interleaved important to balance context lengths across workers
            logging.info(f"--> Benchmark Dataset example input: {self.tokenizer.decode(requests[0], skip_special_tokens=False)}")
            requests = [list(x) for x in distribute(self.data_parallel_size, requests)]
            # Filter out any sublists that may be empty or contain empty data
            requests = [req for req in requests if req and all(sub_req for sub_req in req)]
            print(f"Number of tasks created: {len(requests)}")
            inputs = ((self.model_args, sampling_params, req) for req in requests)
            object_refs = [run_inference_one_model.remote(*x) for x in inputs]
            results = ray.get(object_refs)
            # Invoke ray.shutdown() to prevent hang-ups if subsequent calls required.
            ray.shutdown()
            # flatten results
            flatten_result = undistribute(results)
            logging.info(f"--> Benchmark Dataset example output: {self.tokenizer.decode(flatten_result[0].outputs[0].token_ids, skip_special_tokens=False)}")
            return flatten_result

        else:  # single process/GPU
            logging.info(f"--> Benchmark Dataset example input: {self.tokenizer.decode(requests[0], skip_special_tokens=False)}")
            if self.lora_request is not None:
                outputs = self.model.generate(
                    prompt_token_ids=requests,
                    sampling_params=sampling_params,
                    use_tqdm=True if self.batch_size == "auto" else False,
                    lora_request=self.lora_request,
                )
            else:
                outputs = self.model.generate(
                    prompt_token_ids=requests,
                    sampling_params=sampling_params,
                    use_tqdm=True if self.batch_size == "auto" else False,
                )
            # take the first output batch's first output and print
            logging.info(f"--> Benchmark Dataset example output: {self.tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)}")
            return outputs
