import os
from pathlib import Path
import re
import json
import copy
from typing import List
# don't try to download datasets from internet, it takes too long to fail and fallback to cache
os.environ['HF_DATASETS_OFFLINE'] = "1"
# set the NCCL timeout to 36000 seconds (60 * 10 minutes) to debug nccl
os.environ["NCCL_TIMEOUT"] = "36000"
# when using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD``
#     with the sharding process group being intra-node and the
#     replication process group being inter-node, setting
#     ``NCCL_CROSS_NIC=1`` can help improve the all-reduce times over
#     the replication process group for some cluster setups.
# os.environ["NCCL_CROSS_NIC"] = "1"
# uncomment the following two to print and debug the NCCL based Distribution Training issues
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["NCCL_DEBUG"] = "INFO"
os.environ["WANDB_DISABLED"] = "true"  # disable wandb, only enable it when GPU cluster of training is set up with https access
import fire

# apply triton kernel customizations, it is particularly useful for llama 3 models as vocab size is 13k+
from controlllm.utils.triton_kernels import apply_model_customizations
apply_model_customizations()

import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import StoppingCriteria, StoppingCriteriaList
from accelerate.utils import is_xpu_available
from vllm import LLM, SamplingParams

from controlllm.utils import setup_utils
from controlllm.utils.config_utils import Configs
from controlllm.utils.loading_utils import ModelLoader
from controlllm.utils.model_expander import ModelExpander
from controlllm.configs.datasets import AbstractDataset
from controlllm.data.utils import tokenize_dialog


def main(**kwargs):
    inference_engine = InferenceEngine(**kwargs)
    # Ensure all processes reach this point before proceeding
    torch.distributed.barrier()
    if inference_engine.configs.setup_config.rank == 0:
        inference_engine.run_inference()


class InferenceEngine:
    """
    Inference engine for the chat completion
    """
    def __init__(
        self,
        max_new_tokens: int = 800,  # The maximum numbers of tokens to generate
        prompt_file: str = "/home/jobuser/controlllm/inference/mock_data/chats.json",
        do_sample: bool = True,  # Whether to use sampling; use greedy decoding otherwise.
        use_cache: bool = True,  # Whether the model should use the past last key/values attentions to speed up decoding.
        top_p: float = 1.0,  # If set to float < 1, only the most probable tokens with probabilities adding up to top_p or higher are kept.
        temperature: float = 0.1,  # The value used to modulate the next token probabilities.
        top_k: int = 50,  # The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,  # Exponential penalty to the length, used with beam-based generation.
        stop_tokens: List[str] = ["<|eot_id|>", "<|end_of_text|>"],  # Per llama 3 chat template, use </s> for mistral/mixtral/llama 2
        output_dir: str = "",  # change this to the output directory of the model you want to load if use_vllm is False, and resume_checkpoint_folder for check checkpoint to test
        resume_checkpoint_folder: str = None,  # "checkpoint-3", change 3 to the global step of the checkpoint you want to load, None to respect resume_from_latest
        enable_fsdp: bool = False,  # Enable FullyShardedDataParallel for testing the model with FSDP, only works with native transformers: use_vllm=False
        use_vllm: bool = True,  # Use vllm instead of native transformers if True, set to False to use output_dir and resume_checkpoint_folder, else vllm_model_path
        vllm_model_path: str = "",  # Model path for vllm
        gpu_memory_utilization: float = 0.7,  # GPU memory utilization for vllm
        max_model_len: int = 8192,  # Maximum model length for vllm
        **kwargs  # Accepts any additional keyword arguments
    ):
        # Assign input parameters to class attributes
        self.max_new_tokens = max_new_tokens
        self.prompt_file = prompt_file
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.top_p = top_p
        self.temperature = temperature
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.stop_tokens = stop_tokens
        self.use_vllm = use_vllm
        self.vllm_model_path = vllm_model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.kwargs = kwargs

        # Load the default configuration parameters and update them with the command line arguments
        self.configs = Configs(**kwargs)
        self.configs.model_loading_config.torch_dtype = 'bf16'
        self.configs.model_loading_config.__post_init__()
        self.configs.train_config.enable_fsdp = enable_fsdp
        if output_dir:
            self.configs.train_config.output_dir = output_dir
            self.configs.train_config.resume_checkpoint_folder = resume_checkpoint_folder
        else:
            print("output_dir is required for native transformers but not provided, set use_vllm to True to use vllm model")
        # Sharded weights does not work with kv cache
        if self.configs.train_config.enable_fsdp or self.configs.train_config.enable_deepspeed:
            self.configs.model_loading_config.use_cache = self.use_cache = False
        else:
            self.configs.model_loading_config.use_cache = self.use_cache

        # Set it up to enable runing and debugging in cpu, gpu or distributed mode
        setup_utils.init_distributed_mode(self.configs.setup_config)

        # Load the model
        if self.use_vllm:
            self.load_vllm()
        else:
            self.load_native()

        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        self.dialogs = self.read_dialogs_from_file(prompt_file)
        self.ori_dialogs = copy.deepcopy(self.dialogs)
        print(f"Read {len(self.dialogs)} dialogs from {prompt_file}...")
        print("\n==================================\n")

        if self.configs.setup_config.rank != 0:
            print("Running in mock mode, only rank 0 will run inference")

    def run_inference(self):
        """
        Run the inference loop.
        """
        dialog = []
        while True:
            user_prompt = input("Enter next user prompt (press Enter to exit, type 'system:...' to add system instruction, type '1', '2', etc to execute pre-defined dialogs, type 'reset' to reset the dialog, type 'exit' to exit): \n")

            # if user_prompt is a number, execute the pre-defined dialog
            if user_prompt.isdigit():
                i = int(user_prompt)
                if len(self.dialogs) > i:
                    dialog = self.dialogs[i]
                    print(f"[Logging] Executing pre-defined dialog {i}")
                    for idx, message in enumerate(dialog):
                        if message["role"] == "user":
                            print(f"User: {message['content']}")
                        elif message["role"] == "assistant":
                            print(f"Assistant: {message['content']}")
                        elif message["role"] == "system":
                            print(f"System: {message['content']}")
                    if dialog[-1]["role"] == "assistant":
                        continue
                else:
                    print(f"[Logging] Dialog {i} not found, continue...")
                    continue
            elif user_prompt == "reset":
                dialog = []
                self.dialogs = copy.deepcopy(self.ori_dialogs)
                continue
            elif user_prompt == "exit":
                break
            elif user_prompt == "" or user_prompt is None:
                continue
            else:
                if not user_prompt.startswith("system:"):
                    user_prompt = ''.join(["user:", user_prompt])

                message = self.to_dialog(user_prompt)

                # if user entered system message, don't do inference, continue to let user enter user message
                if message["role"] == "system":
                    continue
                else:  # user role
                    dialog.append(message)

            # AbstractDataset config is not used in inference, make sure AbstractDataset().chat_template is right and AbstractDataset().pretrain = False
            batch = tokenize_dialog(dialog, self.tokenizer, AbstractDataset(dataset="dummy"), inference=True)

            with torch.no_grad():
                if self.use_vllm:
                    generated_texts = self.generate_vllm(batch)
                    output_text = generated_texts[0][0]
                else:
                    outputs = self.generate_transformer(batch)
                    output = outputs[0].cpu().numpy()
                    # output begins with the input tokens, so we skip them
                    output = output[len(batch["input_ids"][0]):]
                    output_text = self.model_loader.tokenizer.decode(output, skip_special_tokens=True)
                if self.configs.setup_config.rank == 0:
                    if output_text.startswith("{") and output_text.endswith("}"):
                        try:
                            output_json = json.loads(output_text)
                            print(f"Assistant:\n{json.dumps(output_json, indent=4)}")

                            # Walk through each item in JSON and check for HTML content
                            for key, value in output_json.items():
                                if isinstance(value, str) and ('<' in value and '>' in value):
                                    print(f"{key}: {self.format_html_content(value)}\n")

                        except json.JSONDecodeError as e:
                            print(f"Assistant:\n{output_text}")
                    else:
                        print(f"Assistant:\n{output_text}")
                    output_text = ''.join(["assistant: ", output_text])
                    message = self.to_dialog(output_text)
                    dialog.append(message)
                    print("\n==================================\n")

    # Generate the response
    def generate_transformer(self, batch):
        """
        Generate the response using the native transformers model.
        """
        # Ensure the model and its inputs are on the same device, add one more dimention to batch values such as batch["input_ids"] which is list of token ids
        def move_to_device(batch):
            for key in batch.keys():
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
                # unsqueeze the tensor to add batch dimension
                if len(batch[key].shape) == 1:
                    batch[key] = batch[key].unsqueeze(0)
                if self.configs.train_config.enable_fsdp or self.configs.train_config.enable_deepspeed:
                    batch[key] = batch[key].to(self.configs.setup_config.local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to(f"xpu:{self.configs.setup_config.local_rank}")
                    else:
                        batch[key] = batch[key].to(f"cuda:{self.configs.setup_config.local_rank}")
        move_to_device(batch)

        if self.configs.train_config.enable_fsdp or self.configs.train_config.enable_deepspeed:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # Note: We need to use the FSDP.summon_full_params context manager here because the generate function
            # does not seem to gather the weights for the LM head. This solution works because the tied weights of the LM head
            # are in the root FSDP module, and are summoned by the below context manager. See https://github.com/pytorch/pytorch/issues/100069
            # for more info.
            # Note: We use recurse=False here so that we only summon full params for the LM head, not the entire model.
            with FSDP.summon_full_params(self.model_loader.model, writeback=False, recurse=False):  # TODO: support deepspeed
                return self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pad_token_id=self.model_loader.tokenizer.pad_token_id,
                        max_new_tokens=self.max_new_tokens,
                        stopping_criteria=self.stopping_criteria,
                        do_sample=self.do_sample,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        use_cache=self.use_cache,
                        top_k=self.top_k,
                        repetition_penalty=self.repetition_penalty,
                        length_penalty=self.length_penalty,
                        synced_gpus=self.configs.setup_config.world_size > 1,
                        **self.kwargs
                    )
        else:
            return self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pad_token_id=self.model_loader.tokenizer.pad_token_id,
                    max_new_tokens=self.max_new_tokens,
                    stopping_criteria=self.stopping_criteria,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    use_cache=self.use_cache,
                    top_k=self.top_k,
                    repetition_penalty=self.repetition_penalty,
                    length_penalty=self.length_penalty,
                    synced_gpus=self.configs.setup_config.world_size > 1,
                    **self.kwargs
                )

    def generate_vllm(self, batch, use_tqdm=False):
        """
        Generate the response using VLLM model.
        """
        # Add one more dimention to list of token ids in batch["input_ids"]
        prompt_token_ids = [batch["input_ids"]]

        # Generate the response from prompt_token_ids
        outputs = self.model.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=self.sampling_params,
            use_tqdm=use_tqdm)
        generated_texts = []

        # Extract the generated texts from the outputs
        for output in outputs:
            generated_texts.append([output.outputs[i].text for i in range(len(output.outputs))])

        return generated_texts

    def load_native(self):
        """
        Load the native transformers model.
        """
        # Which base model it was trained from
        self.model_checkpoint_path = get_last_checkpoint(self.configs.train_config.output_dir) if self.configs.train_config.resume_checkpoint_folder is None else Path(self.configs.train_config.output_dir) / self.configs.train_config.resume_checkpoint_folder
        if self.model_checkpoint_path is None:
            raise ValueError(f"No checkpoint found in {self.configs.train_config.output_dir}, it should have different checkpoint folders like 'checkpoint-1', 'checkpoint-2', ...")
        pretrained_model_name_or_path = ModelLoader.get_trained_from_model_name_or_path(self.model_checkpoint_path)
        if pretrained_model_name_or_path is None:
            raise ValueError(f"Could not find the 'trained_from' in the config of checkpoint: {self.model_checkpoint_path}")
        self.configs.model_loading_config.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.configs.tokenizer_loading_config.pretrained_model_name_or_path = pretrained_model_name_or_path

        # How many expansion layers were added to the model and which merge method was used
        self.configs.model_loading_config = ModelExpander.restore_expansion_configs(self.model_checkpoint_path, self.configs.model_loading_config)

        # Load the pre-trained model and tokenizer
        self.model_loader = ModelLoader(self.configs)

        # Setting `pad_token_id` to `eos_token_id` for open-end generation. 
        self.model_loader.tokenizer.pad_token_id = self.model_loader.tokenizer.eos_token_id

        # Enable model on eval mode
        self.model_loader.model.eval()

        # Prepare the model and tokenizer for generation
        self.model = self.model_loader.model
        self.tokenizer = self.model_loader.tokenizer

        # Prepare stop token ids for stop criteria in generation
        self.stop_token_ids = self._get_stop_token_ids(self.model_loader.tokenizer)

        # Define the stopping criteria
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_token_ids: List[int]):
                self.stop_token_ids = stop_token_ids

            def __len__(self):
                return len(self.stop_token_ids)

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
                device = input_ids.device  # Get the device of the input_ids tensor
                stop_conditions = torch.zeros(input_ids.size()[:-1], dtype=torch.bool, device=device)  # Ensure stop_conditions is on the same device
                for stop_id in self.stop_token_ids:
                    stop_conditions = torch.logical_or(stop_conditions, input_ids[..., -1] == stop_id)
                return stop_conditions

        self.stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids=self.stop_token_ids)])

    def load_vllm(self):
        """
        Load the VLLM model.
        """
        if not self.vllm_model_path:
            raise ValueError("vLLM model path is required for vLLM inference")
        # TODO: enhance register_expansion_classes to support vLLM ModelRegistry
        print(f"Registering the expanded model classes with new model architecture from {self.vllm_model_path}")
        ModelExpander.register_expansion_classes(self.vllm_model_path)

        # Load the vLLM model with single GPU and slightly lower gpu_memory_utilization to avoid OOM
        self.model = LLM(model=self.vllm_model_path, trust_remote_code=True, gpu_memory_utilization=self.gpu_memory_utilization, max_model_len=self.max_model_len, tensor_parallel_size=self.configs.setup_config.world_size, **self.kwargs)
        self.tokenizer = self.model.get_tokenizer()
        self.stop_token_ids = self._get_stop_token_ids(self.model.get_tokenizer())
        self.sampling_params = SamplingParams(n=1, top_p=self.top_p, top_k=self.top_k, temperature=self.temperature, repetition_penalty=self.repetition_penalty, \
                                              length_penalty=self.length_penalty, max_tokens=self.max_new_tokens, stop_token_ids=self.stop_token_ids, seed=self.configs.setup_config.seed)


    def read_dialogs_from_file(self, file_path):
        with open(file_path, 'r') as file:
            dialogs = json.load(file)
        return dialogs

    def to_dialog(self, user_prompt):
        """
        Convert user prompt to dialog format.
        """
        message = None
        if user_prompt.startswith("system:"):
            content = user_prompt.replace("system:", "")
            print(f"Adding system instruction: {content}")
            message = {
                "role": "system",
                "content": content,
            }
        elif user_prompt.startswith("user:"):
            content = user_prompt.replace("user:", "")
            message = {
                "role": "user",
                "content": content,
            }
        elif user_prompt.startswith("assistant:"):
            content = user_prompt.replace("assistant:", "")
            message = {
                "role": "assistant",
                "content": content,
            }
        else:
            print(f"Invalid input: {content}")
        return message

    def format_html_content(self, html):
        """
        Format HTML content for display in the console.
        """
        # Normalize whitespace
        html = ' '.join(html.split())
        # Convert <strong> tags to text indicators, ensuring they stay inline
        html = re.sub(r'<strong>\s*(.*?)\s*</strong>', '[BOLD]\\1[/BOLD]', html)
        # Insert a newline before each <p> and <ul> for readability
        html = re.sub(r'<p>\s*(.*?)\s*</p>', '\n\\1\n', html)
        html = re.sub(r'<ul>\s*(.*?)\s*</ul>', '\n\\1\n', html)
        # Convert <li> tags to format list items with a dash, and ensure they start on a new line
        html = re.sub(r'<li>\s*(.*?)\s*</li>', '\n- \\1', html)
        # Remove all remaining HTML tags (if any)
        html = re.sub(r'<[^>]*>', '', html)
        # Ensure consistent newline handling, strip excessive whitespace
        html = re.sub(r'\n+', '\n', html).strip()
        # Optionally, trim leading newlines from sections
        html = re.sub(r'\n\s+', '\n', html)
        return html

    def _get_stop_token_ids(self, tokenizer):
        # Prepare stop token ids for stop criteria in generation
        stop_token_ids = []
        # special eos_token_id or eot_token_id(llama3) to stop the generation
        for stop_token in self.stop_tokens:
            if stop_token in tokenizer.get_vocab():
                stop_token_ids.append(tokenizer.encode(stop_token)[0])
        print(f"[Logging] Stop token ids: {stop_token_ids}")
        return stop_token_ids


if __name__ == "__main__":
    fire.Fire(main)
