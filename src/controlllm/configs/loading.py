from typing import Optional
import torch
import json
from dataclasses import asdict, dataclass
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


@dataclass
class TokenizerLoadingConfig:
    # tokenizer path separated from model path as checkpoint saving may not have tokenizer
    # pretrained_model_name_or_path: str = "/shared/public/sharing/hawei/Meta-Llama-3-8B"
    # pretrained_model_name_or_path: str = "/shared/public/models/Mistral-7B"
    # pretrained_model_name_or_path: str = "/shared/public/sharing/controlllm/models/llama3-8b-hf-checkpoint"
    # pretrained_model_name_or_path: str = "/shared/public/sharing/controlllm/models/base_model/llama3-10b-hf-checkpoint-no-eu"
    pretrained_model_name_or_path: str = "/shared/public/sharing/hawei/Meta-Llama-3-8B-Instruct"
    # pretrained_model_name_or_path: str = "/shared/public/sharing/controlllm/models/mistral-7b-hf-checkpoint"
    model_max_length: int = 131072
    truncate: bool = False
    truncation_side: str = "right"
    # decoder-only architecture: for correct generation results, please set `padding_side='left'` when initializing the tokenizer.
    padding_side: str = "left"

    # default special tokens, used if tokenizer does not include that token
    default_pad_token = "<PAD>"  # default to llama3's pad token, consider setting it to the right pad token if the model is not llama3
    default_eos_token = "</s>"
    default_bos_token = "<s>"
    default_unk_token = "<unk>"


@dataclass
class ModelLoadingConfig:
    # Model loading arguments for AutoModelForCausalLM.from_pretrained
    # Important note: argument has to be typed to take effect as that is how dataclasses's asdict works

    # loading model from the checkpoint
    # pretrained_model_name_or_path: str = "/shared/public/sharing/hawei/Meta-Llama-3-8B"
    # pretrained_model_name_or_path: str = "/shared/public/models/Mistral-7B"
    # pretrained_model_name_or_path: str = "/shared/public/sharing/controlllm/models/llama3-8b-hf-checkpoint"
    # pretrained_model_name_or_path: str = "/shared/public/sharing/controlllm/models/base_model/llama3-10b-hf-checkpoint-no-eu"
    pretrained_model_name_or_path: str = "/shared/public/sharing/hawei/Meta-Llama-3-8B-Instruct"
    # pretrained_model_name_or_path: str = "/shared/public/sharing/controlllm/models/mistral-7b-hf-checkpoint"
    # Load the model in quantized 8-bit mode with bitsbytes, training with quantization needs to be enabled in TrainConfig
    load_in_8bit: Optional[bool] = False
    load_in_4bit: Optional[bool] = False
    device_map: Optional[str] = None  # keep it None instead of auto to let fsdp or deepspeed do the sharding
    torch_dtype: str = "fp32"  # load the model in fp16, bf16 or fp32
    attn_implementation: str = "flash_attention_2"  # flash_attention_2, sdpa, eager or None
    # tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model
    low_cpu_mem_usage: bool = False  # note: transforms DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`, set it to False in that case
    use_cache: bool = False  # whether to use cache for the model loading
    output_attentions: bool = False  # whether to return attentions in the forward pass, note that it works only when attn_implementation == "eager" or "sdpa" falling back to eager
    # the ratio between the intended max sequence length and the model’s original max sequence length
    rope_factor = 8  # set to None to disable, scaling factor for the ROPE (argument has to be typed to take effect in as_dict, purposely untyped to allow post init), disable means to take the original model's factor
    rope_theta = 5e5  # set to None to disable, theta value for the ROPE, used when rope_factor is not None (argument has to be typed to take effect in as_dict, purposely untyped to allow post init), disable means to take the original model's theta, highly recommended to use the original model's theta
    # whether to strictly enforce that the keys in the state_dict of the model match the keys returned by the model's state_dict() function
    ignore_mismatched_sizes: bool = False
    trust_remote_code: bool = True  # whether to trust the remote code when loading the model

    # number of expansion layers to add to the model, set num_exp_layers to 0 to disable expansion
    num_ori_layers = 32
    num_exp_layers = 8  # has to be even number and num_ori_layers is divisible by num_exp_layers
    expand_type = "concat"  # concat or stack or hybrid
    merge_method = "lerp"  # slerp, lerp, dlerp, dlerpin, moe, proj or prog, used when expand_type is concat or hybrid
    # interpolation_factor - used to interpolate between the outputs of two layers default to 0.5 = sigmoid(0), when set to None, means no interpolation
    interpolation_factor = 0
    interpolation_loss_alpha = 1  # scale to add additional divergence loss, set to 0 to disable it, set to -1 to be dynamic - use alpha=interp_weight, used when expand_type is concat or hybrid
    interpolation_loss_type = "mse"  # 'cosine', 'dot', 'mse', or 'attention', used when expand_type is concat or hybrid. 'attention' needs output_attentions=True.
    interpolation_norm_alpha = 0  # scale to relax the layer norm of expanded layer, set to 0 to disable it, set to -1 to be dynamic - use alpha=interp_weight, set to 1 to disable layer norm, used when expand_type is concat or hybrid
    freeze_interpolation_factor = True  # False to make it trainable, used when merge_method is slerp or lerp. For dlerp, it freezes the bias term if True.
    freeze_ori_layers = True  # freeze the original transformer layers
    freeze_ori_non_layers = True  # freeze the original none transformer layers

    def __post_init__(self, train_config=None):
        if train_config:
            self.load_in_8bit = True if train_config.quantization else self.load_in_8bit
            self.device_map = "auto" if train_config.quantization else self.device_map
            # Use cache is incompatible w/ gradient checkpointing
            if self.use_cache and train_config.gradient_checkpointing:
                self.use_cache = False
            # Use cache is incompatible w/ fsdp
            if self.use_cache and train_config.enable_fsdp:
                self.use_cache = False
            # transforms DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`, set it to False in that case
            if train_config.enable_deepspeed and hasattr(train_config, 'deepspeed') and train_config.deepspeed and "z3" in train_config.deepspeed:
                self.low_cpu_mem_usage = False

        # When using Flash Attention 2 via attn_implementation="flash_attention_2", don’t pass torch_dtype to the from_pretrained class method
        # and use Automatic Mixed-Precision training. When using Trainer, it is simply specifying either fp16 or bf16 to True. 
        # Otherwise, make sure you are using torch.autocast. This is required because the Flash Attention only support fp16 and bf16 data type.
        # Note: be careful on setting to pure_fp16 or pure_bf16, as it may lead to loss instability, reference:
        # https://twitter.com/rohanpaul_ai/status/1738829166269907079
        # https://github.com/huggingface/transformers/pull/28142
        # note: added torch.float16 and torch.bfloat16 in if because there is a second time post_init call from config_utils.py
        if self.attn_implementation == "flash_attention_2" and self.torch_dtype != "fp16" and self.torch_dtype != "bf16" \
            and self.torch_dtype != torch.float16 and self.torch_dtype != torch.bfloat16:
            self.torch_dtype = None
        else:
            # Convert dtype to torch dtype
            dtype_mapping = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            if type(self.torch_dtype) == str:
                try:
                    self.torch_dtype = dtype_mapping[self.torch_dtype]
                except KeyError:
                    raise ValueError(f"Model dtype not supported {self.torch_dtype}")

        # Currently RoPE supports two scaling strategies: linear and dynamic
        # See the following thread for more information on how these scaling strategies behave: 
        # https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/
        if self.rope_factor:
            # follow https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/preparing_model
            self.rope_scaling = {"type": "linear", "factor": self.rope_factor}

        # Set the decoder layer based on the model name, this is not passed in AutoModelForCausalLM.from_pretrained on purpose
        if "llama" in self.pretrained_model_name_or_path.lower():
            # for llama 3.2 [LlamaDecoderLayer, MllamaSelfAttentionDecoderLayer,MllamaVisionEncoderLayer,MllamaCrossAttentionDecoderLayer]
            self.decoder_layer = [LlamaDecoderLayer]
        elif "mistral" in self.pretrained_model_name_or_path.lower():
            self.decoder_layer = [MistralDecoderLayer]
        elif "mixtral" in self.pretrained_model_name_or_path.lower():
            self.decoder_layer = [MixtralDecoderLayer]
        elif "qwen" in self.pretrained_model_name_or_path.lower():
            self.decoder_layer = [Qwen2DecoderLayer]
        else:
            raise ValueError("Model not supported")

    @staticmethod
    def serialize_model_config(config):
        config_dict = asdict(config)
        # Convert specific fields if necessary
        config_dict['torch_dtype'] = str(config.torch_dtype) if config.torch_dtype else None
        return json.dumps(config_dict, cls=EnhancedJSONEncoder)

    @staticmethod
    def deserialize_model_config(json_str):
        config_dict = json.loads(json_str)
        # Convert fields back to their intended types
        dtype_mapping = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
        }
        config_dict['torch_dtype'] = dtype_mapping.get(config_dict['torch_dtype'], None)
        # Re-create the dataclass instance with the dictionary
        return ModelLoadingConfig(**config_dict)


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.dtype):
            return str(obj)  # Convert torch.dtype to string
        return json.JSONEncoder.default(self, obj)
