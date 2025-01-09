import re
import logging
from packaging import version
from datetime import timedelta
from typing import Dict, List, Optional, Union
from transformers import PreTrainedModel

import torch
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs
)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from controlllm.utils.model_expander import ModelExpander


eval_logger = utils.eval_logger


@register_model("llama_plus")
class LlamaPlusWrapper(HFLM):
    def __init__(
        self,
        pretrained: Union[str, PreTrainedModel] = "",
        device: Optional[str] = "cuda",
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        *args,
        **kwargs,
    ) -> None:
        """
        A llm eval harness wrapper of Llama Plus model.
        """
        # Extract pretrained as a string of where the model checkpoint exists
        if isinstance(pretrained, PreTrainedModel):
            model_name_or_path = pretrained.name_or_path
            pretrained.eval()  # Ensure model is in eval mode
        elif isinstance(pretrained, str):
            model_name_or_path = pretrained
        else:
            raise ValueError(f"Invalid pretrained argument: {model_name_or_path}, expected str or PreTrainedModel.")

        if not model_name_or_path:
            raise ValueError("Could not find a valid model name or path from the provided model argument. {model_name_or_path=}.")
        logging.info(f"Registering the expanded model classes with new model architecture from {model_name_or_path}")
        ModelExpander.register_expansion_classes(model_name_or_path)
        logging.info(f"Initializing Llama Plus model from {model_name_or_path} to prepare for benchmarking...")

        super().__init__(
            pretrained=pretrained,
            device=device,
            parallelize=parallelize,
            autogptq=autogptq,
            *args,
            **kwargs,
        )

        # default huggingface.py in lm-evaluation-harness supports only single-process call, custom the logic here to support multi-process by "accelerate launch" command
        if isinstance(pretrained, PreTrainedModel):
            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            # using one process with no model parallelism
            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:  # Parallelism managed by accelerate
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )

            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        Generate text from the model until a stop token is reached.

        args:
            requests: The list of instances.
            disable_tqdm: Whether to disable tqdm.
        """
        return super().generate_until(
            requests=requests,
            disable_tqdm=disable_tqdm,
        )

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

    def _create_model(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        """
        Initializes an HF-compatible PreTrainedModel for llama repro from scratch
        """
        super()._create_model(
            pretrained=pretrained,
            **kwargs,
        )

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        """
        Generate text from the model.

        args:
            context: The input_ids.
            max_length: The maximum length of the generated text.
            stop: The stop token(s) to stop generation at.
            generation_kwargs: Additional generation kwargs.
        """
        logging.info(f"--> Benchmark Dataset example input: {self.tokenizer.decode(context[0], skip_special_tokens=False)}")
        results = super(LlamaPlusWrapper, self)._model_generate(
            context=context,
            max_length=max_length,
            stop=stop,
            **generation_kwargs,
        )
        logging.info(f"--> Benchmark Dataset example output: {self.tokenizer.decode(results[0][context.shape[1] :], skip_special_tokens=False)}")
        return results
