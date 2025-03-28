'''note that we want to keep this python file independent from rest of framework such as loading_utils.py so the community can use it without the need of controlllm framework'''
from functools import partial
import yaml
import shutil
import inspect
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import ModuleList, Module, Linear, Sigmoid, Parameter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from dataclasses import asdict
from transformers import AutoModel, AutoConfig, GenerationConfig, AutoModelForCausalLM
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, CONFIG_MAPPING, PretrainedConfig, PreTrainedModel
from vllm import ModelRegistry

from controlllm.utils.custom_sentence_transformers import CustomSentenceTransformer as SentenceTransformer
from sentence_transformers.util import is_sentence_transformer_model

from controlllm.configs.loading import ModelLoadingConfig


class ModelExpander:
    def __init__(self, model_loading_config: str, resume_checkpoint_path: str = None):
        """
        Initialize the ModelExpander with the model loading configuration.
        """
        # Deserialize the JSON string back to the ModelLoadingConfig object
        self.model_loading_config: ModelLoadingConfig = ModelLoadingConfig.deserialize_model_config(model_loading_config)

        # Restore the expansion configurations in the model config to model loading config. Note that expansion configs are not passed along as part of deserialize_model_config and serialize_model_config
        # This is to handle special case of user does not config ./configs/loading.py consistently with what was used to train the resumed checkpoint
        if resume_checkpoint_path:
            self.model_loading_config = self.restore_expansion_configs(resume_checkpoint_path, self.model_loading_config)

        self.num_ori_layers = self.model_loading_config.num_ori_layers
        self.num_exp_layers = self.model_loading_config.num_exp_layers
        self.expand_type = self.model_loading_config.expand_type
        self.merge_method = self.model_loading_config.merge_method
        self.interpolation_loss_alpha = self.model_loading_config.interpolation_loss_alpha
        self.interpolation_loss_type = self.model_loading_config.interpolation_loss_type
        self.interpolation_norm_alpha = self.model_loading_config.interpolation_norm_alpha
        self.interpolation_factor = self.model_loading_config.interpolation_factor
        self.freeze_interpolation_factor = self.model_loading_config.freeze_interpolation_factor

        # Register the custom model classes for expansion
        print(f"Model expansion: Registering the expanded model classes with new model architecture from {self.model_loading_config.pretrained_model_name_or_path}")
        self.register_expansion_classes(self.model_loading_config.pretrained_model_name_or_path)

        # Load the configuration from a pretrained model
        print(f"Model expansion: Loading pretrained model configuration from {self.model_loading_config.pretrained_model_name_or_path}")

        self.config = AutoConfig.from_pretrained(self.model_loading_config.pretrained_model_name_or_path)
        self.generation_config = GenerationConfig.from_pretrained(self.model_loading_config.pretrained_model_name_or_path)

        # Dynamically modify the configuration based on expansion requirements, note that concat expansion does not increase number of layers
        if self.config.num_hidden_layers == self.num_ori_layers and self.config.architectures[0] != self.CausalLMExpansion.__name__.lower() and not hasattr(self.config, 'expansion'):  # not yet expanded
            # Initialize the model with the updated configuration
            print(f"Model expansion: Loading base model from {self.model_loading_config.pretrained_model_name_or_path}")
            self.model: Union[AutoModelForCausalLM, SentenceTransformer] = self._from_pretrained(**asdict(self.model_loading_config))
        else:
            self.model = None

    def _from_pretrained(self, **kwargs):
        # TODO: support fsdp_cpu_ram_efficient_loading for SentenceTransformer
        if is_sentence_transformer_model(kwargs["pretrained_model_name_or_path"]):  # support local SentenceTransformer models only(download it first since no token passed for authentication of huggingface model hub)
            pretrained_model_name_or_path = kwargs.pop("pretrained_model_name_or_path")
            use_cache, output_attentions = kwargs.pop("use_cache"), kwargs.pop("output_attentions")  # remove the use_cache and output_attentions from kwargs because here we load the auto lm model not causal lm model
            st_model = SentenceTransformer(model_name_or_path=pretrained_model_name_or_path, model_kwargs=kwargs)
            # SentenceTransformer uses AutoModel.from_pretrained internally which does not support these attributes(difference from AutoModelForCausalLM.from_pretrained), so pop and set these attributes back
            st_model.config.use_cache, st_model.config.output_attentions = use_cache, output_attentions
            return st_model
        else:
            return AutoModelForCausalLM.from_pretrained(**kwargs)

    def expand_layers(self):
        """
        Expand the model based on the specified expansion type.
        """
        print(f"Model expansion: Expanding model with {self.num_exp_layers} layers using {self.expand_type} expand type")
        if self.expand_type == "stack":
            self.expand_layers_stack()
        elif self.expand_type == "concat":
            self.expand_layers_concat()
        elif self.expand_type == "hybrid":
            self.expand_layers_hybrid()
        else:
            raise ValueError(f"Invalid model expansion type: {self.expand_type}")

    @classmethod
    def merge_layers(cls, model, model_path_or_name=None):
        """
        Merge the expanded layers back into the original layers after training.
        This is the reverse operation of expand_layers.
        """
        expand_type = model.config.expansion["expand_type"]
        if expand_type == "stack":
            cls.merge_layers_stack(model)
        elif expand_type == "concat":
            cls.merge_layers_concat(model, model_path_or_name)
        elif expand_type == "hybrid":
            cls.merge_layers_hybrid(model, model_path_or_name)
        else:
            raise ValueError(f"Invalid model expansion type: {cls.expand_type}")

    def expand_layers_stack(self):
        """
        Expand the model by stacking the new layers on top of the original layers.
        """
        num_new_layers = self.num_ori_layers + self.num_exp_layers
        print(f"Model expansion: Original number of layers: {self.num_ori_layers}, New number of layers: {num_new_layers}")

        if num_new_layers <= self.num_ori_layers or len(self.model.model.layers) == num_new_layers:
            print("Model expansion: No expansion needed or invalid expansion configuration.")
            return

        split = max(int(self.num_ori_layers // (self.num_exp_layers)), 1)
        expanded_layers = ModuleList()
        self.to_be_freezed = set()

        original_layers = self.model.model.layers
        block_cnt = 0
        for i in range(self.num_ori_layers):
            expanded_layers.append(original_layers[i])
            expanded_layers[-1].layer_idx = block_cnt
            self.to_be_freezed.add(block_cnt)
            block_cnt += 1

            if (i + 1) % split == 0 and block_cnt < num_new_layers:
                print(f"Model expansion: Inserting new layer at {block_cnt} by copying from {i}")
                # Create a new layer by directly invoking its constructor
                # Note that instead of using deepcopy, the code manually replicates the layer by instantiating a new layer of the same type and copying over its parameters. 
                # This ensures that all aspects of the layer's setup (like parameter tensors) are correctly duplicated without unintentionally copying internal state or references that deepcopy might mishandle.
                # When using deepcopy in the context of a model that utilizes DeepSpeed's ZeRO-3, there will be empty tensor issues because deepcopy does not respect the partitioning of model parameters across different GPUs
                new_layer = type(original_layers[i])(self.model.config, block_cnt)
                new_layer.load_state_dict(original_layers[i].state_dict())
                new_layer.layer_idx = block_cnt
                # zero out both self attn and mlp weights
                self._zero_out_weights(new_layer, ['o_proj', 'down_proj'])
                # self._reset_weights(new_layer, ['input_layernorm', 'post_attention_layernorm'])
                new_layer.requires_grad_(True)
                expanded_layers.append(new_layer)
                block_cnt += 1

        self.model.model.layers = expanded_layers

        # Update the config:
        print(f"Model expansion: Updating model configuration with expand type: {self.expand_type}, expanded layers: {[i for i in range(num_new_layers) if i not in self.to_be_freezed]}")
        self.model.config.num_hidden_layers += self.num_exp_layers
        # Additional expansion configs
        self._set_expansion_configs()

    def expand_layers_concat(self):
        """
        Expand the model by adding the new layers as side car of the original layers.
        """
        split = max(int(self.num_ori_layers // self.num_exp_layers), 1)
        expanded_layers = ModuleList()
        self.to_be_freezed = set()

        original_layers = self.model.model.layers
        for layer_idx in range(self.num_ori_layers):
            expanded_layers.append(original_layers[layer_idx])

            if (layer_idx + 1) % split == 0:
                print(f"Model expansion: Create side car of the new layer at {layer_idx} by copying from pretrained layers {layer_idx}")
                # self.interpolation_norm_alpha, self.interpolation_factor are only passed in for initialization, will be reassigned by MergeLayer
                expanded_layer = self.ExpandedDecoderLayer(self.model.config, layer_idx, self.interpolation_norm_alpha, self.interpolation_factor, self.freeze_interpolation_factor)
                expanded_layer.load_state_dict(original_layers[layer_idx].state_dict(), strict=False)  # strict=False becasue new parameter interpolation_factor is added
                # self._reset_weights(expanded_layer, ['input_layernorm', 'post_attention_layernorm'])
                expanded_layer.requires_grad_(True)

                # Follow ControlNet paper's implementation and make the new layer a side car of the original layer and do a add operation
                # Take the last appended layer and add the merge layer
                pretrained_layer = expanded_layers[-1]
                # Create a new merge layer idea borrowed from ControlNet implementation
                # Adding a parallel layer(expanded_layer) that will have its outputs combined at a later point in MergeLayer
                expanded_layers[-1] = self._create_merged_layer(pretrained_layer, expanded_layer, layer_idx)
            else:
                self.to_be_freezed.add(layer_idx)

        self.model.model.layers = expanded_layers

        # Update the config:
        print(f"Model expansion: Updating model configuration with expand type: {self.expand_type} - {self.merge_method}, expanded layers: {[i for i in range(self.num_ori_layers) if i not in self.to_be_freezed]}")
        self.model.config.architectures = [f"{self.model.config.architectures[0]}Expansion"]
        # Additional expansion configs
        self._set_expansion_configs()
        self.model.config = self.CausalLMExpansionConfig.from_dict(self.model.config.to_dict())

        # expanded model is still with the original model class, update the model class to the expanded model class
        if is_sentence_transformer_model(self.model_loading_config.pretrained_model_name_or_path):
            self.model.model.config = self.model.config
            self.model.model.__class__ = self.ModelExpansion
            self.model.generation_config = self.generation_config
        else:
            self.model.__class__ = self.CausalLMExpansion
            self.model.generation_config = self.generation_config

    def expand_layers_hybrid(self):
        """
        For every other expanded layer:
            Expand the model by adding the new layers as side car of the original layers.
            Expand the model by stacking the new layers on top of the original layers.
        """
        num_new_layers = self.num_ori_layers + self.num_exp_layers // 2
        print(f"Model expansion: Original number of layers: {self.num_ori_layers}, New number of layers: {num_new_layers}")

        if num_new_layers <= self.num_ori_layers or len(self.model.model.layers) == num_new_layers:
            print("Model expansion: No expansion needed or invalid expansion configuration.")
            return

        split = max(int(self.num_ori_layers // self.num_exp_layers), 1)
        expanded_layers = ModuleList()
        self.to_be_freezed = set()
        original_layers = self.model.model.layers
        block_cnt = 0
        method_index = 0
        expand_methods = ['stack', 'concat']

        for i in range(self.num_ori_layers):
            expanded_layers.append(original_layers[i])
            expanded_layers[-1].layer_idx = block_cnt
            block_cnt += 1

            if (i + 1) % split == 0 and block_cnt <= num_new_layers:
                expand_method = expand_methods[method_index % 2]
                method_index += 1

                if expand_method == 'stack':
                    self.to_be_freezed.add(block_cnt - 1)
                    print(f"Model expansion: Inserting new layer at {block_cnt} by copying from {i} (stack)")
                    new_layer = type(original_layers[i])(self.model.config, block_cnt)
                    new_layer.load_state_dict(original_layers[i].state_dict())
                    new_layer.layer_idx = block_cnt
                    self._zero_out_weights(new_layer, ['o_proj', 'down_proj'])
                    new_layer.requires_grad_(True)
                    expanded_layers.append(new_layer)
                    block_cnt += 1

                elif expand_method == 'concat':
                    print(f"Model expansion: Creating side car of the new layer at {block_cnt - 1} by copying from pretrained layer {i} (concat)")
                    # self.interpolation_norm_alpha, self.interpolation_factor are only passed in for initialization, will be reassigned by MergeLayer
                    expanded_layer = self.ExpandedDecoderLayer(self.model.config, block_cnt - 1, self.interpolation_norm_alpha, self.interpolation_factor, self.freeze_interpolation_factor)
                    expanded_layer.load_state_dict(original_layers[i].state_dict(), strict=False)  # strict=False becasue new parameter interpolation_factor is added
                    expanded_layer.requires_grad_(True)

                    pretrained_layer = expanded_layers[-1]
                    pretrained_layer.layer_idx = block_cnt - 1

                    expanded_layers[-1] = self._create_merged_layer(pretrained_layer, expanded_layer, block_cnt - 1)
            else:
                self.to_be_freezed.add(block_cnt - 1)

        self.model.model.layers = expanded_layers

        # Update the config:
        print(f"Model expansion: Updating model configuration with expand type: {self.expand_type}, expanded layers: {[i for i in range(len(expanded_layers)) if i not in self.to_be_freezed]}")
        assert block_cnt - len(original_layers) == self.num_exp_layers // 2, "Number of expanded layers does not match the expected number"
        self.model.config.num_hidden_layers += self.num_exp_layers // 2
        # Additional expansion configs
        self._set_expansion_configs()
        self.model.config.architectures = [f"{self.model.config.architectures[0]}Expansion"]
        self.model.config = self.CausalLMExpansionConfig.from_dict(self.model.config.to_dict())

        # expanded model is still with the original model class, update the model class to the expanded model class
        if is_sentence_transformer_model(self.model_loading_config.pretrained_model_name_or_path):
            self.model.model.config = self.model.config
            self.model.model.__class__ = self.ModelExpansion
            self.model.generation_config = self.generation_config
        else:
            self.model.__class__ = self.CausalLMExpansion
            self.model.generation_config = self.generation_config

    def _create_merged_layer(self, pretrained_layer, expanded_layer, layer_idx):
        params = (pretrained_layer, expanded_layer, layer_idx, self.interpolation_factor, self.interpolation_loss_alpha, self.interpolation_loss_type, self.freeze_interpolation_factor)
        if self.merge_method == "slerp":
            return MergeLayerSlerp(*params)
        elif self.merge_method == "dlerp":
            return MergeLayerDlerp(*params)
        elif self.merge_method == "dlerpin":
            return MergeLayerDlerpIn(*params)
        elif self.merge_method == "moe":
            return MergeLayerMoE(*params)
        elif self.merge_method == "lerp":
            return MergeLayerLerp(*params)
        elif self.merge_method == "proj":
            return MergeLayerProj(*params)
        elif self.merge_method == "prog":
            return MergeLayerProg(*params)
        else:
            raise ValueError(f"Invalid merge method: {self.merge_method}, supported: slerp, dlerp, dlerpin, lerp, proj, prog, moe")

    def _set_expansion_configs(self):
        """
        Set the expansion configurations in the model config.
        """
        expansion_config = {
            "expand_type": self.model_loading_config.expand_type,
            "merge_method": self.model_loading_config.merge_method,
            "freezed_layers": list(self.to_be_freezed),  # save the freezed layers for reference of continuous training
            "expanded_from": self.model_loading_config.pretrained_model_name_or_path,
            "num_ori_layers": self.model_loading_config.num_ori_layers,
            "num_exp_layers": self.model_loading_config.num_exp_layers,
            "interpolation_factor": self.model_loading_config.interpolation_factor,  # used to interpolate between the outputs of two layers, default to 0.0
            "interpolation_loss_alpha": self.model_loading_config.interpolation_loss_alpha,  # enable divergent loss for the expanded layers
            "interpolation_loss_type": self.model_loading_config.interpolation_loss_type,  # 'cosine', 'dot', 'mse', or 'attention'
            "interpolation_norm_alpha": self.model_loading_config.interpolation_norm_alpha,  # scale to relax the layer norm
            "freeze_interpolation_factor": self.model_loading_config.freeze_interpolation_factor,  # change to True to make it trainable, used when merge_method is slerp or lerp
            "freeze_ori_layers": self.model_loading_config.freeze_ori_layers,  # freeze the original transformer layers
            "freeze_ori_non_layers": self.model_loading_config.freeze_ori_non_layers  # freeze the original none transformer layers
        }

        self.model.config.expansion = expansion_config
        self.model.config.trained_from = self.model_loading_config.pretrained_model_name_or_path  # save the original model name for reference, this will be overwritten by training script of SFT or RFHF

        # Remove auto_map from the config to avoid loading custom model class by dynamic module loading.
        # this is to make sure model loaded with model class from transformers' AutoModel, AutoModelForCausalLM, etc. ./configs/loading.py needs that to wrap the model with FSDP
        if hasattr(self.model.config, 'auto_map'):
            del self.model.config.auto_map
            logging.warning("Model expansion: Removed auto_map from the model config to avoid loading custom model class by dynamic module loading. Disable it and reconfigure ./configs/loading.py if this is necessary.")

    @classmethod
    def restore_expansion_configs(cls, model_path_or_name: str, model_loading_config: ModelLoadingConfig) -> ModelLoadingConfig:
        """
        Restore the expansion configurations in the model config to model loading config.
        """
        expansion_config = cls.get_expansion_configs(model_path_or_name)
        if expansion_config is None:
            logging.warning(f"Model expansion: Expansion configs does not exist in model config of {model_path_or_name}, skipping restore.")
            return model_loading_config
        else:
            model_loading_config.expand_type = expansion_config.get("expand_type")
            model_loading_config.merge_method = expansion_config.get("merge_method")
            # purposely commented out: do not restore pretrained model name from expanded_from, pretrained_model_name_or_path in modle_loading_config might be the expanded model name during continuous pretraining phase not the original model name from huggerface
            # model_loading_config.pretrained_model_name_or_path = expansion_config.get("expanded_from")
            model_loading_config.num_ori_layers = expansion_config.get("num_ori_layers")
            model_loading_config.num_exp_layers = expansion_config.get("num_exp_layers")
            model_loading_config.interpolation_factor = expansion_config.get("interpolation_factor", 0)
            model_loading_config.interpolation_loss_alpha = expansion_config.get("interpolation_loss_alpha", 0)
            model_loading_config.interpolation_loss_type = expansion_config.get("interpolation_loss_type", "cosine")
            model_loading_config.interpolation_norm_alpha = expansion_config.get("interpolation_norm_alpha", 0)
            model_loading_config.freeze_interpolation_factor = expansion_config.get("freeze_interpolation_factor", False)
            model_loading_config.freeze_ori_layers = expansion_config.get("freeze_ori_layers", True)
            model_loading_config.freeze_ori_non_layers = expansion_config.get("freeze_ori_non_layers", True)

            return model_loading_config

    # Identity transformer blocks: zeroing out the weights of down_proj in feedforward and o_proj in the self attention
    @classmethod
    def _zero_out_weights(cls, module, target_names):
        """
        Recursively zero out specific weights and biases in a PyTorch module.

        Args:
            module (torch.nn.Module): The module to modify.
            target_names (list of str): Names of submodules or weights to be zeroed out.
        """
        for name, sub_module in module.named_children():
            if any(target_name in name for target_name in target_names):
                if hasattr(sub_module, 'weight'):
                    sub_module.weight.data.fill_(0.0)
                    print(f"Zeroed weight in {name}")
                if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                    sub_module.bias.data.fill_(0.0)
                    print(f"Zeroed bias in {name}")
            cls._zero_out_weights(sub_module, target_names)  # Recurse into submodules

    # Layer Normalization Reset: initialize weights to one (as norm layers typically standardize their inputs to zero mean and unit variance) and biases to zero,
    # which is supported by the original Layer Normalization paper
    @classmethod
    def _reset_weights(cls, module, target_names):
        """
        Recursively reset specific weights and biases in a PyTorch module.

        Args:
            module (torch.nn.Module): The module to modify.
            target_names (list of str): Names of submodules or weights to be reset.
        """
        for name, sub_module in module.named_children():
            if any(target_name in name for target_name in target_names):
                if hasattr(sub_module, 'weight'):
                    sub_module.weight.data.fill_(1.0)
                    print(f"Reset weight in {name}")
                if hasattr(sub_module, 'bias') and sub_module.bias is not None:
                    sub_module.bias.data.fill_(0.0)
                    print(f"Reset bias in {name}")
            cls._reset_weights(sub_module, target_names)  # Recurse into submodules

    def save_model(self, output_path):
        print(f"Model expansion: Saving expanded model to {output_path}")
        if isinstance(self.model, SentenceTransformer):
            self.model.save_pretrained(output_path)
        else:
            self.model.save_pretrained(output_path, save_config=True)

        # save the module_file if there is any, e.g. modeling_qwen.py when config.json has custom module by auto_map.
        # "auto_map": {
        #     "AutoModel": "modeling_qwen.Qwen2Model",
        #     "AutoModel": "modeling_qwen.Qwen2ForCausalLM",
        #     "AutoModelForSequenceClassification": "modeling_qwen.Qwen2ForSequenceClassification",
        # },
        if "auto_map" in self.model.config.__dict__:
            for class_reference in self.model.config.auto_map.values():
                module_file, class_name = class_reference.split(".")
                logging.info(f"Model expansion: Custom model class found in config -{module_file}.{class_name}")
                custom_model_file = Path(self.model_loading_config.pretrained_model_name_or_path) / f"{module_file}.py"
                output_model_file = Path(output_path)/f"{module_file}.py"
                if not output_model_file.exists():
                    logging.info(f"Model expansion: Copying custom model file from {custom_model_file} to {output_path}")
                    shutil.copy(custom_model_file, output_model_file)
                else:
                    logging.info(f"Model expansion: Custom model file {custom_model_file} already exists in {output_path}, skipping copying.")

    # Define this as a class method to continue training of previously expanded model with mixing frozen and non-frozen
    @classmethod
    def freeze_layers(cls, model, freezed_layers_path):
        """
        Freeze the layers based on the model.config.expansion.freezed_layers.
        """
        expansion_config = cls.get_expansion_configs(freezed_layers_path)
        if expansion_config is None or "freezed_layers" not in expansion_config:
            logging.warning(f"Model expansion: Freezed layers does not exist in model config of {freezed_layers_path}, skipping freeze.")
            return

        to_be_freezed = cls.get_expansion_configs(freezed_layers_path)["freezed_layers"]

        # Freeze the original layers as specified in the json file
        logging.info(f"Freezing original layers at {to_be_freezed}")
        for i, layer in enumerate(model.model.layers):
            if i in to_be_freezed:
                for param in layer.parameters():
                    param.requires_grad_(False)

        # If expand type is concat, freeze the pretrained layers
        if model.config.expansion["expand_type"] == "concat" or model.config.expansion["expand_type"] == "hybrid":
            logging.info(f"Model expansion: expand type is {model.config.expansion['expand_type']} - {model.config.expansion['merge_method']}. Freezing pretrained layers in merged layer in addition to original layers.")
            for i, layer in enumerate(model.model.layers):
                if isinstance(layer, MergeLayer):
                    logging.info(f"Model expansion: Freezing pretrained layer as part of merged layers at {i}")
                    for param in layer.pretrained_layer.parameters():
                        param.requires_grad_(False)

    # Define this as a class method to continue training of previously expanded model with mixing frozen and non-frozen
    @classmethod
    def freeze_none_layers(cls, model):
        """
        Freeze the none layers of the model including embedding tokens, norm and lm_head.
        """
        # Get the set of all submodules in model.model.layers
        layer_submodules = set()
        for layer in model.model.layers:
            layer_submodules.update(layer.modules())

        # Iterate over all modules in the model and freeze those not in layer_submodules
        for name, module in model.named_modules():
            # If the module has trainable parameters and is not part of the layer submodules
            if module not in layer_submodules and module != model.model:
                # Check if this module is a parent container of any layer submodule
                is_container_of_submodule = any(submodule in module.modules() for submodule in layer_submodules)

                if not is_container_of_submodule:
                    logging.info(f"Freezing none transformer layer - module: {name}")
                    for param in module.parameters():
                        param.requires_grad_(False)

    @classmethod
    def register_expansion_classes(cls, model_path_or_name, use_vllm=False):
        """
        Register the custom model classes for expansion in huggingface transformers.
        This is useful to make it compatible with load_pretrained and save_pretrained methods.

        Args:
            model_path_or_name (str): The path or name of the model to expand or the expanded model.
        """
        # If already registered, skip to save time, TODO: commented out to support running eval without use_vllm and running benchmark with use_vllm in one run
        if "causallmexpansion" in CONFIG_MAPPING:
            print(f"Model expansion regsitering: Custom model {CONFIG_MAPPING['causallmexpansion']} is already registered in transformers, skipping.")
            return

        expansion_config = cls.get_expansion_configs(model_path_or_name)

        if expansion_config:
            assert "expanded_from" in expansion_config, "expanded_from is not found in the expansion config"
            hf_model_path_or_name = expansion_config["expanded_from"]
            print(f"Model expansion regsitering: Base Model {model_path_or_name} is expanded from {hf_model_path_or_name}")
        else:
            hf_model_path_or_name = model_path_or_name
            print(f"Model expansion regsitering: Base Model {model_path_or_name} is with original model class in huggerface")

        # Load the original huggingface configuration to inherit the base model configuration and class for transformers
        print(f"Model expansion regsitering: Loading original hf model configuration from {hf_model_path_or_name} for registering new/expanded model class")
        ori_hf_config = AutoConfig.from_pretrained(hf_model_path_or_name)
        if use_vllm:
            model_class = ModelRegistry.resolve_model_cls(ori_hf_config.architectures)[0]  # returns (model_cls, arch), so take first one
        else:
            model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(ori_hf_config)]

        config_class = type(ori_hf_config)

        # 1: Inherit and extend the original LayerNorm class

        # get the module where the model class is defined
        model_module = inspect.getmodule(model_class)
        # find the RMSNorm class
        input_layernorm_class = None
        for name, obj in inspect.getmembers(model_module, inspect.isclass):
            if issubclass(obj, torch.nn.Module) and name.endswith('RMSNorm'):
                input_layernorm_class = obj
                break
        if input_layernorm_class is not None:
            print(f"The layer norm class is: {input_layernorm_class}")
        else:
            raise ValueError("Layer norm class not found.")

        # Design a new layer norm class to enable relaxed layer norm
        class ExpandedLayerNorm(input_layernorm_class):
            def __init__(self, hidden_size, eps=1e-6, interpolation_norm_alpha=0, interpolation_factor=0, freeze_interpolation_factor=True):
                super().__init__(hidden_size, eps=eps)
                self.interpolation_norm_alpha = interpolation_norm_alpha
                self.device, self.dtype = next(self.parameters()).device, next(self.parameters()).dtype

                if self.interpolation_norm_alpha == -1:  # -1 indicates dynamic interpolation weight based on interpolation factor, so introduce a parameter for interpolation factor
                    if freeze_interpolation_factor:
                        # Make it a frozen parameter to ensure it is part of self.named_parameters() and compatible with the weight-loading process
                        self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=self.dtype, device=self.device), requires_grad=False)  # Set requires_grad to False to freeze it
                    else:
                        # Make interpolation_factor a trainable parameter to adjust the interpolation factor
                        self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=self.dtype, device=self.device))

            def forward(self, hidden_states, *args, **kwargs):
                if self.interpolation_norm_alpha == 0:  # No interpolation
                    return super().forward(hidden_states, *args, **kwargs)
                else:  # either fixed or dynamic interpolation for relaxed layer norm
                    orig_hidden_states = hidden_states
                    norm_hidden_states = super().forward(hidden_states, *args, **kwargs)
                    interp_norm_weight = self.get_interpolation_norm_weight()
                    # Expand the interpolation weight to the same shape as the hidden states
                    interp_norm_weight = interp_norm_weight.expand_as(orig_hidden_states) if isinstance(interp_norm_weight, torch.Tensor) else interp_norm_weight
                    if isinstance(norm_hidden_states, Tuple):  # for vllm, the output could be a tuple of hidden states and residuals
                        return interp_norm_weight * orig_hidden_states + (1 - interp_norm_weight) * norm_hidden_states[0], norm_hidden_states[1]
                    else:
                        return interp_norm_weight * orig_hidden_states + (1 - interp_norm_weight) * norm_hidden_states

            def get_interpolation_norm_weight(self):
                if self.interpolation_norm_alpha == 0:
                    shifted_interp_weight = 0
                elif self.interpolation_norm_alpha == -1:
                    # Dynamic interpolation weight based on the interpolation factor
                    interp_weight = torch.sigmoid(self.interpolation_factor)
                    # Transform interp_weight from range 0-1 (starting at 0.5) to range 0-1 (starting at 0)
                    shifted_interp_weight = 2 * interp_weight - 1
                    # Ensure the value is within the range 0 to 1
                    shifted_interp_weight = torch.clamp(shifted_interp_weight, 0, 1)
                else:
                    shifted_interp_weight = self.interpolation_norm_alpha

                return shifted_interp_weight

        cls.ExpandedLayerNorm = ExpandedLayerNorm

        # 2: Inherit and extend the original DecoderLayer class

        # get the module where the model class is defined
        model_module = inspect.getmodule(model_class)
        # find the DecoderLayer class
        decoder_layer_class = None
        for name, obj in inspect.getmembers(model_module, inspect.isclass):
            if issubclass(obj, torch.nn.Module) and name.endswith('DecoderLayer'):
                decoder_layer_class = obj
                break
        if decoder_layer_class is not None:
            print(f"The decoder layer class is: {decoder_layer_class}")
        else:
            raise ValueError("Decoder layer class not found.")

        class ExpandedDecoderLayer(decoder_layer_class):
            def __init__(self, config, layer_idx, interpolation_norm_alpha=0, interpolation_factor=0, freeze_interpolation_factor=True, **kwargs):
                if use_vllm:
                    super().__init__(config, **kwargs)
                else:
                    super().__init__(config, layer_idx)

                # overwrite the original layer norm with the expanded layer norm class
                self.input_layernorm = cls.ExpandedLayerNorm(config.hidden_size, config.rms_norm_eps, interpolation_norm_alpha, interpolation_factor, freeze_interpolation_factor)
                self.post_attention_layernorm = cls.ExpandedLayerNorm(config.hidden_size, config.rms_norm_eps, interpolation_norm_alpha, interpolation_factor, freeze_interpolation_factor)

        cls.ExpandedDecoderLayer = ExpandedDecoderLayer

        # 3: Inherit and extend the original model config class

        class CausalLMExpansionConfig(config_class):
            model_type = "causallmexpansion"

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.expansion = kwargs.pop("expansion", {})

        cls.CausalLMExpansionConfig = CausalLMExpansionConfig

        # 4: Inherit and expand the original auto lm model class
        # get the module where the auto lm model class is defined
        prefix = model_class.__name__.replace("ForCausalLM", "")
        model_module = inspect.getmodule(model_class)
        # find the AutoModel class
        auto_model_class = None
        for name, obj in inspect.getmembers(model_module, inspect.isclass):
            if issubclass(obj, torch.nn.Module) and name.endswith('Model') and name.startswith(prefix):
                auto_model_class = obj
                break
        if auto_model_class is not None:
            print(f"The auto model class is: {auto_model_class}")
        else:
            raise ValueError("Auto model class not found.")

        # this is to register it with new config class
        class ModelExpansion(auto_model_class):
            config_class = cls.CausalLMExpansionConfig

            # vllm load model by model class with additional parameters so introduce *args, **kwargs here, refer to https://github.com/vllm-project/vllm/blob/fc6c27462614924dca90898ef762d6c56c0874ba/vllm/model_executor/model_loader/loader.py#L160
            def __init__(self, config: PretrainedConfig, *args, **kwargs):
                if use_vllm:
                    super().__init__(config, *args, **kwargs)
                    # Customize the forward method for vLLM model, this is to make sure kv cache update is done properly as expanded layer and pretrained layer are branched out
                    # self.model.forward = llama_vllm__forward
                else:
                    super().__init__(config)

                # Expand/Customize the pretrained model architecture before loading the pretrained and expanded weights
                cls._expand_model_for_register(config, self, use_vllm, *args, **kwargs)

        # 5: Inherit and expand the original causal lm model class

        if use_vllm and expansion_config:  # monkey patch the forward method of vllm model
            from vllm.model_executor.models.llama import LlamaModel
            LlamaModel.forward = llama_vllm__forward
            from vllm.worker.cache_engine import CacheEngine
            # vllm_cache_engine___allocate_kv_cache_defaulted = partial(vllm_cache_engine___allocate_kv_cache, freezed_layers=expansion_config["freezed_layers"])
            # Create a wrapper to include the default freezed_layers parameter similar to partial, note that partial does not work for class method

            def freezed_layers(self, num_blocks: int, device: str):
                # Create a partial function with self and freezed_layers inside the wrapper
                custom_allocate_kv_cache = partial(
                    vllm_cache_engine___allocate_kv_cache,
                    self=self,
                    freezed_layers=expansion_config["freezed_layers"]
                )
                return custom_allocate_kv_cache(num_blocks=num_blocks, device=device)

            CacheEngine._allocate_kv_cache = freezed_layers

        class CausalLMExpansion(model_class):
            config_class = cls.CausalLMExpansionConfig

            # vllm load model by model class with additional parameters so introduce *args, **kwargs here, refer to https://github.com/vllm-project/vllm/blob/fc6c27462614924dca90898ef762d6c56c0874ba/vllm/model_executor/model_loader/loader.py#L160
            def __init__(self, config: PretrainedConfig, *args, **kwargs):
                if use_vllm:
                    super().__init__(config, *args, **kwargs)
                    # Customize the forward method for vLLM model, this is to make sure kv cache update is done properly as expanded layer and pretrained layer are branched out
                    # self.model.forward = llama_vllm__forward
                else:
                    super().__init__(config)

                # Make self.model which is already instantiated to be the expanded model class ModelExpansion
                self.model.__class__ = ModelExpansion

                # Expand/Customize the pretrained model architecture before loading the pretrained and expanded weights
                cls._expand_model_for_register(config, self.model, use_vllm, *args, **kwargs)

        cls.CausalLMExpansion = CausalLMExpansion
        cls.ModelExpansion = ModelExpansion

        # 6: Register the new model class with transformers along with the new model config class

        # To update CONFIG_MAPPING_NAMES["causallmexpansion"] = "CausalLMExpansionConfig"
        AutoConfig.register(CausalLMExpansion.__name__.lower(), cls.CausalLMExpansionConfig, exist_ok=True)

        # Register custom model to transformers: https://huggingface.co/docs/transformers/en/custom_models, to update MODEL_FOR_CAUSAL_LM_MAPPING[cls.CausalLMExpansionConfig] = cls.CausalLMExpansion
        AutoModelForCausalLM.register(cls.CausalLMExpansionConfig, cls.CausalLMExpansion, exist_ok=True)

        # Register AutoModel with the new config class
        AutoModel.register(cls.CausalLMExpansionConfig, cls.ModelExpansion, exist_ok=True)

        # Register custom model to vLLM: https://docs.vllm.ai/en/v0.5.5/models/adding_model.html
        ModelRegistry.register_model(CausalLMExpansion.__name__, cls.CausalLMExpansion)
        print(f"Model expansion regsitering: Registered new model class {CausalLMExpansion.__name__} with transformers and vLLM")

    @classmethod
    def _expand_model_for_register(cls, config, model, use_vllm=False, *args, **kwargs):
            if config.expansion["expand_type"] == "concat":
                # Expand the model by adding new layers as sidecar of the original layers by configuration
                new_layers = []
                for layer_idx, layer in enumerate(model.layers):
                    if layer_idx not in model.config.expansion["freezed_layers"]:
                        print(f"Model expansion regsitering - concat: Concatenating new layer at layer {layer_idx}")

                        def get_value_or_default(config_dict, key, default):
                            return config_dict.get(key) if config_dict.get(key) is not None else default

                        # get_value_or_default(config.expansion, "interpolation_factor", None) is to support interpolation_factor being None in the config: no interpolation
                        interpolation_factor, interpolation_loss_alpha, interpolation_loss_type, interpolation_norm_alpha, freeze_interpolation_factor = get_value_or_default(config.expansion, "interpolation_factor", None), get_value_or_default(config.expansion, "interpolation_loss_alpha", 0), get_value_or_default(config.expansion, "interpolation_loss_type", "cosine"), get_value_or_default(config.expansion, "interpolation_norm_alpha", 0), get_value_or_default(config.expansion, "freeze_interpolation_factor", False)

                        if use_vllm:
                            # Create a new layer of the same class with the same configuration
                            expanded_layer = cls.ExpandedDecoderLayer(config, layer_idx, interpolation_norm_alpha, interpolation_factor, freeze_interpolation_factor, cache_config=kwargs.get("cache_config"), quant_config=kwargs.get("quant_config"), prefix=kwargs.get("prefix"))
                        else:
                            # Create a new layer of the same class with the same configuration, note that interpolation_norm_alpha, interpolation_factor are only passed in here for initialization, will be reassigned in the MergeLayer init method
                            expanded_layer = cls.ExpandedDecoderLayer(config, layer_idx, interpolation_norm_alpha, interpolation_factor, freeze_interpolation_factor)

                        # Manually copy parameters from the original layer to the new layer
                        expanded_layer.load_state_dict(layer.state_dict(), strict=False)  # strict=False becasue new parameter interpolation_factor is added
                        expanded_layer.requires_grad_(True)

                        # layer.requires_grad_(False)  # freeze the original layer typically, but not needed here since it will be determined by config.expansion["freeze_ori_layers"]
                        # Insert a MergeLayer which handles the logic of combining outputs
                        params = (layer, expanded_layer, layer_idx, interpolation_factor, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)
                        if config.expansion["merge_method"] == "slerp":
                            merged_layer = MergeLayerSlerp(*params)
                        elif config.expansion["merge_method"] == "dlerp":
                            merged_layer = MergeLayerDlerp(*params)
                        elif config.expansion["merge_method"] == "dlerpin":
                            merged_layer = MergeLayerDlerpIn(*params)
                        elif config.expansion["merge_method"] == "moe":
                            merged_layer = MergeLayerMoE(*params)
                        elif config.expansion["merge_method"] == "lerp":
                            merged_layer = MergeLayerLerp(*params)
                        elif config.expansion["merge_method"] == "proj":
                            merged_layer = MergeLayerProj(*params)
                        elif config.expansion["merge_method"] == "prog":
                            merged_layer = MergeLayerProg(*params)
                        else:
                            raise ValueError(f"Invalid merge method: {config.expansion['merge_method']}, supported: slerp, dlerp, dlerpin, lerp, proj, prog, moe")
                        new_layers.append(merged_layer)
                    else:
                        # For frozen layers, simply reuse the original layer
                        new_layers.append(layer)

                # Replace the old layers with the new merged layers
                model.layers = ModuleList(new_layers)

            elif config.expansion["expand_type"] == "hybrid":
                # Expand the model by alternating between stacking new layers and adding them as sidecars
                new_layers = []
                method_index = 0
                expand_methods = ['stack', 'concat']

                for layer_idx, layer in enumerate(model.layers):
                    if layer_idx not in config.expansion["freezed_layers"]:
                        expand_method = expand_methods[method_index % 2]
                        method_index += 1

                        # half of the expanded layers are stacked and the other half are concatenated
                        if expand_method == 'stack':
                            # new layers are already stacked by the changed model config(model.config.num_hidden_layers), so no need to expand here
                            print(f"Model expansion regsitering - hybrid: Keeping stacked layer {layer_idx} unfrozen")
                            new_layers.append(layer)
                        elif expand_method == 'concat':
                            # Concat expansion: merge original layer with a new expanded layer
                            print(f"Model expansion regsitering - hybrid: Concatenating new layer at layer {layer_idx}")

                            # get_value_or_default(config.expansion, "interpolation_factor", None) is to support interpolation_factor being None in the config: no interpolation
                            interpolation_factor, interpolation_loss_alpha, interpolation_loss_type, interpolation_norm_alpha, freeze_interpolation_factor = config.expansion.get("interpolation_factor", None), config.expansion.get("interpolation_loss_alpha", 0), config.expansion.get("interpolation_loss_type", "cosine"), config.expansion.get("interpolation_norm_alpha", 0), config.expansion.get("freeze_interpolation_factor", False)

                            # Create a new layer of the same class with the same configuration, note that interpolation_norm_alpha, interpolation_factor are only passed in here for initialization, will be reassigned in the MergeLayer init method
                            if use_vllm:
                                # Create a new layer of the same class with the same configuration
                                expanded_layer = cls.ExpandedDecoderLayer(config, layer_idx, interpolation_norm_alpha, interpolation_factor, freeze_interpolation_factor, cache_config=kwargs.get("cache_config"), quant_config=kwargs.get("quant_config"), prefix=kwargs.get("prefix"))
                            else:
                                # Create a new layer of the same class with the same configuration, note that interpolation_norm_alpha, interpolation_factor are only passed in here for initialization, will be reassigned in the MergeLayer init method
                                expanded_layer = cls.ExpandedDecoderLayer(config, layer_idx, interpolation_norm_alpha, interpolation_factor, freeze_interpolation_factor)

                            expanded_layer.load_state_dict(layer.state_dict(), strict=False)  # strict=False becasue new parameter interpolation_factor is added
                            expanded_layer.requires_grad_(True)

                            # Insert a MergeLayer which handles the logic of combining outputs
                            # layer.requires_grad_(False)  # freeze the original layer typically, but not needed here since it will be determined by config.expansion["freeze_ori_layers"]
                            params = (layer, expanded_layer, layer_idx, interpolation_factor, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)
                            if config.expansion["merge_method"] == "slerp":
                                merged_layer = MergeLayerSlerp(*params)
                            elif config.expansion["merge_method"] == "dlerp":
                                merged_layer = MergeLayerDlerp(*params)
                            elif config.expansion["merge_method"] == "dlerpin":
                                merged_layer = MergeLayerDlerpIn(*params)
                            elif config.expansion["merge_method"] == "moe":
                                merged_layer = MergeLayerMoE(*params)
                            elif config.expansion["merge_method"] == "lerp":
                                merged_layer = MergeLayerLerp(*params)
                            elif config.expansion["merge_method"] == "proj":
                                merged_layer = MergeLayerProj(*params)
                            elif config.expansion["merge_method"] == "prog":
                                merged_layer = MergeLayerProg(*params)
                            else:
                                raise ValueError(f"Invalid merge method: {config.expansion['merge_method']}, supported: slerp, dlerp, dlerpin, lerp, proj, prog, moe")
                            new_layers.append(merged_layer)

                    else:
                        # For frozen layers, simply reuse the original layer
                        print(f"Model expansion regsitering - hybrid: Keeping original layer {layer_idx} frozen")
                        new_layers.append(layer)

                # Replace the old layers with the new expanded layers
                model.layers = ModuleList(new_layers)
                assert model.config.num_hidden_layers == len(new_layers), "Number of expanded layers does not match the expected number"
            else:
                raise ValueError(f"Invalid expand type: {config.expansion['expand_type']}, supported: concat, hybrid")

    @classmethod
    def get_additional_loss(cls, model, loss) -> torch.Tensor:
        # Initialize the additional loss as a tensor with the same device and dtype as the input loss
        additional_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        count = 0  # Counter for the number of layers with an additional loss

        for layer in model.model.layers:
            if hasattr(layer, 'additional_loss') and layer.additional_loss is not None:
                additional_loss += layer.additional_loss
                count += 1

        if count > 0:
            additional_loss /= count

        return additional_loss

    @classmethod
    def get_interp_weights(cls, model) -> dict:
        interp_factors = {}

        def gather_interp_factors(model):
            for layer in model.model.layers:
                if hasattr(layer, 'interpolation_factor') and layer.interpolation_factor is not None:
                    # Now you can access the full interpolation_factor
                    gathered_interp_factor = layer.interpolation_factor.detach().cpu()
                    interp_factors[layer.layer_idx] = torch.sigmoid(gathered_interp_factor)

        # Check if the model is an instance of FSDP
        if isinstance(model, FSDP):
            with FSDP.summon_full_params(model):
                gather_interp_factors(model)
        else:
            gather_interp_factors(model)

        avg_interp_weight = {}
        for layer in model.model.layers:
            if hasattr(layer, 'avg_interp_weight'):
                # Now you can access the full interpolation_factor
                avg_interp_weight[layer.layer_idx] = layer.avg_interp_weight
                # Note: we reset the avg_interp_weight to 0.0 and num_interp_weight to 0 after reading it
                layer.avg_interp_weight, layer.num_interp_weight = 0.0, 0

        # Optional: Only log on the root rank (rank 0)
        if dist.get_rank() == 0:
            return interp_factors, avg_interp_weight
        else:
            return {}, {}

    @classmethod
    def enable_model_debug(cls, model, metadata, model_logger):
        """
        Plugin the model with debug information for tensorboard visualization.

        model: The model to be debugged.
        metadata: The metadata to be attached to the model layers. e.g. {'labels': ['king-queen', "man-woman"], 'global_step': 0}
        model_logger: The model loger to cumulate the debug information.
        """
        for layer in model.model.layers:
            if isinstance(layer, MergeLayer):
                layer.metadata = metadata
                layer.model_logger = model_logger

    @classmethod
    def get_expansion_configs(cls, model_path_or_name) -> Optional[dict]:
        """
        Get the expansion configurations in the model config.
        """
        if not Path(model_path_or_name).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path_or_name}")

        # Load the config from fsdp_checkpoint_path/config.json into dictionary
        if not (Path(model_path_or_name) / "config.json").exists():
            raise FileNotFoundError(f"config.json not found in {model_path_or_name}")

        with open(Path(model_path_or_name) / "config.json", "r") as f:
            config_dict = yaml.safe_load(f)
        if "expansion" in config_dict:
            return config_dict["expansion"]
        else:
            return None

    @classmethod
    def _load_avg_interp_weights(cls, model_path_or_name: str) -> dict:
        """
        Load avg_interp_weight that is saved during evaluation run for the merged layer.
        Return dict of avg_interp_weight at the specified layer index or None if not found.
        """
        if model_path_or_name is None:
            return None

        if not Path(model_path_or_name).exists():
            print(f"Model path does not exist: {model_path_or_name}")
            return None

        # Load the evaluation result from model_path_or_name/config.json into dictionary
        if not (Path(model_path_or_name) / "evaluation_results.json").exists():
            print(f"evaluation_results.json not found in {model_path_or_name}")
            return None

        with open(Path(model_path_or_name) / "evaluation_results.json", "r") as f:
            eval_result_dict = yaml.safe_load(f)

        eval_avg_interp_weights = eval_result_dict.get("eval_avg_interp_weights", None)
        print(f"Loaded avg_interp_weights from evaluation results: {eval_avg_interp_weights}")

        # key was formated by f"layer_{layer_idx}", removing layer_ to get the layer index as key
        avg_interp_weights = {int(key.split("_")[1]): value for key, value in eval_avg_interp_weights.items()} if eval_avg_interp_weights is not None else None

        return avg_interp_weights

    @classmethod
    def merge_layers_concat(cls, model, model_path_or_name=None):
        """
        Merge the expanded layer back to pretrained layer after the expanded layer is trained.
        This is the reverse operation of expand_layers_concat.
        model_path_or_name is needed to load the avg_interp_weights from the evaluation results for weights of merging back.
        """
        # expect model.config has expand_type and merge_method
        assert hasattr(model.config, "expansion"), "Model config does not have expansion config, skipping merge."
        expand_type, merge_method, num_exp_layers = (model.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("num_exp_layers", None)])
        print(f"Model expansion - merge back: Merging expanded layers back to pretrained layers with expand type: {expand_type} - {merge_method}")
        if expand_type != "concat" or merge_method not in ["slerp", "lerp", "dlerp", 'dlerpin']:
            print(f"Model expansion - merge back: Invalid expand type: {expand_type} or merge method: {merge_method}, skipping merge. Only support concat and slerp/lerp/dlerp/dlerpin.")
            return

        # Load the original huggingface configuration to revert back to the base model configuration and class
        hf_model_path_or_name = model.config.expansion["expanded_from"]
        print(f"Model expansion - merge back: Loading base model configuration from {hf_model_path_or_name} to restore the model confit after merging back")
        ori_hf_config = AutoConfig.from_pretrained(hf_model_path_or_name)
        ori_model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(ori_hf_config)]
        ori_config_class = type(ori_hf_config)

        # Load the avg_interp_weights from the evaluation results for weights of merging back
        if model_path_or_name is not None:
            print(f"Model expansion - merge back: Loading avg_interp_weights from evaluation results for interpolation weight saved in {model_path_or_name}")
            avg_interp_weights = cls._load_avg_interp_weights(model_path_or_name)
        else:
            avg_interp_weights = None

        num_ori_layers = ori_hf_config.num_hidden_layers
        num_freezed_layers = len(model.config.expansion["freezed_layers"])
        assert num_exp_layers == num_ori_layers - num_freezed_layers, "Number of expanded layers from the config does not match the expected number of num_ori_layers - num_freezed_layers"
        split = max(int(num_ori_layers // num_exp_layers), 1)

        freezed_layers = list(model.config.expansion["freezed_layers"])
        weight_merger = WeightMerger(merge_method=merge_method)

        merged_layers = ModuleList()
        expanded_layers = model.model.layers
        for layer_idx in range(num_ori_layers):
            merged_layers.append(expanded_layers[layer_idx])

            if (layer_idx + 1) % split == 0:
                merged_layer = expanded_layers[layer_idx]
                if merge_method == "slerp":
                    assert isinstance(merged_layer, MergeLayerSlerp), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerSlerp"
                elif merge_method == "lerp":
                    assert isinstance(merged_layer, MergeLayerLerp), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerLerp"
                elif merge_method == "dlerp":
                    assert isinstance(merged_layer, MergeLayerDlerp), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerDlerp"
                elif merge_method == "dlerpin":
                    assert isinstance(merged_layer, MergeLayerDlerpIn), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerDlerpIn"
                else:
                    raise ValueError(f"Invalid merge method to merge weights: {merge_method}, supported: slerp, lerp")

                print(f"Model Expansion - merge back: Merging side car of the expanded layer at {layer_idx} to pretrained layer {layer_idx}")
                expanded_layer = merged_layer.expanded_layer
                pretrained_layer = merged_layer.pretrained_layer

                # merge the expanded layer back to the pretrained layer by merging the weights
                if avg_interp_weights is not None and layer_idx in avg_interp_weights:
                    print(f"Model Expansion - merge back: Using avg_interp_weights from evaluation results for interpolation weight, layer: {layer_idx}: {avg_interp_weights[layer_idx]}")
                    interp_weight = avg_interp_weights[layer_idx]
                else:
                    print(f"Model Expansion - merge back: Using interpolation factor from the merge layer of the model for interpolation weight, layer: {layer_idx}: {torch.sigmoid(merged_layer.interpolation_factor)=}")
                    interp_weight = torch.sigmoid(merged_layer.interpolation_factor)
                print(f"Model Expansion - merge back: Layer: {layer_idx} Interpolation factor: {interp_weight}")
                weight_merger.merge_weights(pretrained_layer, expanded_layer, interp_weight)

                # Replace the merged layer with the pretrained layer
                merged_layers[-1] = pretrained_layer
            else:
                assert layer_idx in freezed_layers, f"Model Expansion - merge back: Layer {layer_idx} should be freezed but not in freezed layers: {freezed_layers}"

        model.model.layers = merged_layers

        # update the config:
        print(f"Model expansion - merge back: Updating model configuration with expand type: {expand_type} - {merge_method}, merged layers: {[i for i in range(num_ori_layers) if i not in freezed_layers]}")
        model.config.architectures = ori_hf_config.architectures
        assert model.config.num_hidden_layers == len(model.model.layers), f"Number of hidden layers in the model config ({model.config.num_hidden_layers}) does not match the number of layers in the model ({len(model.model.layers)})."
        model.config = ori_config_class.from_dict(model.config.to_dict())
        # still keep expansion_config: expand_type, expanded_from, merge_method, freezed_layers etc, for reference of being able to trace back on how the model was expanded and trained
        model.config.expansion = model.config.expansion
        model.config.expansion["num_exp_layers"] = -num_exp_layers  # change the num_exp_layers to negative to indicate the model is merged back

        # model is still the expanded model class, we need to update the model class to the original model class
        model.__class__ = ori_model_class

    @classmethod
    def merge_layers_stack(cls, model):
        """
        Merge the expanded layers back into the original layers after training.
        This is the reverse operation of expand_layers_stack.
        """
        # Retrieve expansion config from the model
        assert hasattr(model.config, "expansion"), "Model config does not have expansion config, skipping merge."
        expand_type, merge_method, interpolation_factor, num_exp_layers = (model.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("interpolation_factor", 0.5), ("num_exp_layers", None)])
        num_exp_layers = model.config.expansion["num_exp_layers"]
        print(f"Model expansion - merge back: Merging expanded layers back into original layers with expand type: {expand_type}")
        if expand_type != "stack":
            print(f"Model expansion - merge back: Invalid expand type: {expand_type}, skipping merge. Only support 'stack'.")
            return

        # Load the original HuggingFace configuration to revert back to the base model configuration and class
        hf_model_path_or_name = model.config.expansion["expanded_from"]
        print(f"Model expansion - merge back: Loading base model configuration from {hf_model_path_or_name} to restore the model config after merging back.")
        ori_hf_config = AutoConfig.from_pretrained(hf_model_path_or_name)
        ori_model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(ori_hf_config)]
        ori_config_class = type(ori_hf_config)

        num_ori_layers = ori_hf_config.num_hidden_layers
        num_total_layers = len(model.model.layers)
        assert num_exp_layers == num_total_layers - num_ori_layers, "Number of expanded layers from the config does not match the expected number of num_total_layers - num_ori_layers"
        split = max(int(num_ori_layers // num_exp_layers), 1)

        weight_merger = WeightMerger(merge_method=merge_method)  # change config for needed merge_method regardless of the original merge_method trained with

        merged_layers = ModuleList()
        layers = model.model.layers

        i = 0  # Index in original layers
        block_cnt = 0  # Index in layers
        while block_cnt < len(layers):
            layer = layers[block_cnt]
            block_cnt += 1
            # Check if an expanded layer follows after 'split' layers
            if (i + 1) % split == 0 and block_cnt < len(layers):
                # Next layer is an expanded layer
                expanded_layer = layers[block_cnt]
                print(f"Model expansion - merge back: Merging expanded layer at index {block_cnt} back into original layer {i}")
                # Merge weights of expanded_layer into layer
                interp_weight = torch.sigmoid(torch.tensor(interpolation_factor))  # change config for needed interpolation_factor regardless of the original merge_method trained with
                weight_merger.merge_weights(layer, expanded_layer, interp_weight)
                block_cnt += 1  # Skip the expanded layer
            merged_layers.append(layer)
            i += 1

        model.model.layers = merged_layers

        # Update the configuration
        print(f"Model expansion - merge back: Updating model configuration to original model configuration with {num_ori_layers} layers.")
        model.config.architectures = ori_hf_config.architectures
        model.config.num_hidden_layers = num_ori_layers
        assert model.config.num_hidden_layers == len(model.model.layers), f"Number of hidden layers in the model config ({model.config.num_hidden_layers}) does not match the number of layers in the model ({len(model.model.layers)})."
        model.config = ori_config_class.from_dict(model.config.to_dict())
        # Keep expansion config for reference
        model.config.expansion = model.config.expansion
        model.config.expansion["num_exp_layers"] = -num_exp_layers  # change the num_exp_layers to negative to indicate the model is merged back

        # Update model class to original model class
        model.__class__ = ori_model_class

    @classmethod
    def merge_layers_hybrid(cls, model, model_path_or_name=None, merge_concat_only=True):
        """
        Merge the expanded layers back into the original layers after training.
        This is the reverse operation of expand_layers_hybrid.
        model_path_or_name is needed to load the avg_interp_weights from the evaluation results for weights of merging back.
        Set merge_concat_only to False to merge both stacked and concatenated expanded layers back.
        """
        # Retrieve expansion config from the model
        assert hasattr(model.config, "expansion"), "Model config does not have expansion config, skipping merge."
        expand_type, merge_method, interpolation_factor, num_exp_layers = (model.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("interpolation_factor", 0.5), ("num_exp_layers", None)])
        print(f"Model expansion - merge back: Merging expanded layers back into original layers with expand type: {expand_type}")
        if expand_type != "hybrid":
            print(f"Model expansion - merge back: Invalid expand type: {expand_type}, skipping merge. Only support 'hybrid'.")
            return

        # Load the original HuggingFace configuration to revert back to the base model configuration and class
        hf_model_path_or_name = model.config.expansion["expanded_from"]
        print(f"Model expansion - merge back: Loading base model configuration from {hf_model_path_or_name} to restore the model config after merging back.")
        ori_hf_config = AutoConfig.from_pretrained(hf_model_path_or_name)
        ori_model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(ori_hf_config)]
        ori_config_class = type(ori_hf_config)

        # Load the avg_interp_weights from the evaluation results for weights of merging back
        if model_path_or_name is not None:
            print(f"Model expansion - merge back: Loading avg_interp_weights from evaluation results for interpolation weight saved in {model_path_or_name}")
            avg_interp_weights = cls._load_avg_interp_weights(model_path_or_name)
        else:
            avg_interp_weights = None

        num_ori_layers = ori_hf_config.num_hidden_layers
        num_total_layers = len(model.model.layers)
        # In expand_layers_hybrid, num_new_layers = num_ori_layers + num_exp_layers // 2
        # Therefore, num_exp_layers = (num_total_layers - num_ori_layers) * 2
        num_exp_layers = (num_total_layers - num_ori_layers) * 2
        split = max(int(num_ori_layers // num_exp_layers), 1)

        weight_merger = WeightMerger(merge_method=merge_method)
        merged_layers = ModuleList()
        layers = model.model.layers

        block_cnt = 0  # Index in layers
        method_index = 0
        expand_methods = ['stack', 'concat']
        layer_idx = 0  # Original layer index

        while block_cnt < len(layers):
            if (layer_idx + 1) % split == 0 and method_index < num_exp_layers:
                expand_method = expand_methods[method_index % 2]
                method_index += 1

                if expand_method == 'stack' and (not merge_concat_only):
                    # Stack expansion: merge expanded layer back into the previous layer
                    original_layer = layers[block_cnt]
                    expanded_layer = layers[block_cnt + 1]

                    print(f"Model expansion - merge back: Merging stacked expanded layer at index {block_cnt + 1} back into original layer {layer_idx}")
                    # Merge weights of expanded_layer into original_layer
                    interp_weight = interpolation_factor  # Use provided interpolation_factor
                    weight_merger.merge_weights(original_layer, expanded_layer, interp_weight)
                    merged_layers.append(original_layer)
                    block_cnt += 2  # Skip the expanded layer
                elif expand_method == 'concat':
                    # Concat expansion: merged layer is at block_cnt
                    merged_layer = layers[block_cnt]
                    if merge_method == "slerp":
                        assert isinstance(merged_layer, MergeLayerSlerp), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerSlerp"
                    elif merge_method == "lerp":
                        assert isinstance(merged_layer, MergeLayerLerp), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerLerp"
                    elif merge_method == "dlerp":
                        assert isinstance(merged_layer, MergeLayerDlerp), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerDlerp"
                    elif merge_method == "dlerpin":
                        assert isinstance(merged_layer, MergeLayerDlerpIn), f"Invalid merge layer type: {type(merged_layer)}, expected MergeLayerDlerpIn"
                    else:
                        raise ValueError(f"Invalid merge method to merge weights: {merge_method}, supported: slerp, lerp")

                    print(f"Model Expansion - merge back: Merging concatenated expanded layer at index {block_cnt} to original layer {layer_idx}")
                    expanded_layer = merged_layer.expanded_layer
                    pretrained_layer = merged_layer.pretrained_layer
                    # Merge the expanded layer back into the pretrained layer by merging the weights
                    if avg_interp_weights is not None and layer_idx in avg_interp_weights:
                        print(f"Model Expansion - merge back: Using avg_interp_weights from evaluation results for interpolation weight, layer: {layer_idx}: {avg_interp_weights[layer_idx]}")
                        interp_weight = avg_interp_weights[layer_idx]
                    else:
                        print(f"Model Expansion - merge back: Using interpolation factor from the merge layer of the model for interpolation weight, layer: {layer_idx}: {torch.sigmoid(merged_layer.interpolation_factor)=}")
                        interp_weight = torch.sigmoid(merged_layer.interpolation_factor)
                    print(f"Model Expansion - merge back: Layer: {layer_idx} Interpolation factor: {interp_weight}")
                    weight_merger.merge_weights(pretrained_layer, expanded_layer, interp_weight)
                    # Replace the merged layer with the pretrained layer
                    merged_layers.append(pretrained_layer)
                    block_cnt += 1
            else:
                # Not an expanded layer, keep the layer as is
                merged_layers.append(layers[block_cnt])
                block_cnt += 1

            layer_idx += 1

        model.model.layers = merged_layers

        # Update the configuration
        print(f"Model expansion - merge back: Updating model configuration to original model with {ori_hf_config.architectures}.")
        model.config.architectures = ori_hf_config.architectures
        model.config.num_hidden_layers = num_ori_layers
        assert model.config.num_hidden_layers == len(model.model.layers), f"Number of hidden layers in the model config ({model.config.num_hidden_layers}) does not match the number of layers in the model ({len(model.model.layers)})."
        model.config = ori_config_class.from_dict(model.config.to_dict())
        # Keep expansion config for reference
        model.config.expansion = model.config.expansion
        model.config.expansion["num_exp_layers"] = -num_exp_layers  # Indicate that the model is merged back

        # Update model class to original model class
        model.__class__ = ori_model_class


    @classmethod
    def merge_models(cls, model_a, model_b, interp_weight=0.5) -> PreTrainedModel:
        """
        This is used to merge two checkpoints of the same model with same expand_type.
        Assuming freeze_ori_non_layers is True, the original layers are frozen and the expanded layers are trained.
        """
        # expect model.config has expand_type and merge_method
        assert hasattr(model_a.config, "expansion") and hasattr(model_a.config, "expansion"), "Model config does not have expansion config, skipping merge."
        expand_type_a, merge_method_a, num_exp_layers_a = (model_a.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("num_exp_layers", None)])
        print(f"Model expansion - merging: Merging model a: {expand_type_a} - {merge_method_a}")

        expand_type_b, merge_method_b, num_exp_layers_b = (model_b.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("num_exp_layers", None)])
        print(f"Model expansion - merging: Merging model b: {expand_type_b} - {merge_method_b}")

        assert model_a.config.expansion["expanded_from"] == model_b.config.expansion["expanded_from"], "Model expanded from different base model, skipping merge."
        assert expand_type_a == expand_type_b, "Expand type does not match between model_a and model_b"
        assert merge_method_a == merge_method_b, "Merge method does not match between model_a and model_b"
        assert num_exp_layers_a == num_exp_layers_b, "Number of expanded layers from the config does not match between model_a and model_b"

        freezed_layers = list(model_a.config.expansion["freezed_layers"])
        weight_merger = WeightMerger(merge_method=merge_method_a)

        merged_layers = ModuleList()
        model_a_layers = model_a.model.layers
        model_b_layers = model_b.model.layers
        for layer_idx in range(len(model_a_layers)):
            merged_layers.append(model_a_layers[layer_idx])
            if layer_idx not in freezed_layers:
                print(f"Model expansion - merging: Merging layer: {layer_idx} with interpolation weight: {interp_weight}")
                # merge the weights of the two layers in place from model_b_layers[layer_idx] to model_a_layers[layer_idx]
                weight_merger.merge_weights(model_a_layers[layer_idx], model_b_layers[layer_idx], interp_weight)
        model_a.model.layers = merged_layers

        return model_a


    @classmethod
    def merge_models_concat(cls, model_a, model_b, merge_back=True) -> PreTrainedModel:
        """
        This is used to merge two checkpoints of the same model trained by expansion with expand_type concat.

        If merge_back is True, merge the expanded layer back to pretrained layer after the expanded layer is trained for two models.
        Else, this is the operation to merge two models with the same base model and expansion config while keeping expanded layers.
        """
        # expect model.config has expand_type and merge_method
        assert hasattr(model_a.config, "expansion") and hasattr(model_a.config, "expansion"), "Model config does not have expansion config, skipping merge."
        expand_type_a, merge_method_a, num_exp_layers_a = (model_a.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("num_exp_layers", None)])
        print(f"Model expansion - merge back: Merging expanded layers back to pretrained layers with expand type: {expand_type_a} - {merge_method_a}")
        if expand_type_a != "concat" or merge_method_a not in ["slerp", "lerp", "dlerp", "dlerpin"]:
            print(f"Model expansion - merge back: Invalid expand type: {expand_type_a} or merge method: {merge_method_a}, skipping merge. Only support concat and slerp/lerp/dlerp/dlerpin.")
            return

        expand_type_b, merge_method_b, num_exp_layers_b = (model_b.config.expansion.get(key, default) for key, default in [("expand_type", None), ("merge_method", None), ("num_exp_layers", None)])
        print(f"Model expansion - merge back: Merging expanded layers back to pretrained layers with expand type: {expand_type_b} - {merge_method_b}")
        if expand_type_b != "concat" or merge_method_b not in ["slerp", "lerp", "dlerp", "dlerpin"]:
            print(f"Model expansion - merge back: Invalid expand type: {expand_type_b} or merge method: {merge_method_b}, skipping merge. Only support concat and slerp/lerp/dlerp/dlerpin.")
            return

        assert model_a.config.expansion["expanded_from"] == model_b.config.expansion["expanded_from"], "Model expanded from different base model, skipping merge."
        assert num_exp_layers_a == num_exp_layers_b, "Number of expanded layers from the config does not match between model_a and model_b"

        # Load the original huggingface configuration to revert back to the base model configuration and class
        hf_model_path_or_name = model_a.config.expansion["expanded_from"]
        print(f"Model expansion - merge back: Loading base model configuration from {hf_model_path_or_name} to restore the model confit after merging back")
        ori_hf_config = AutoConfig.from_pretrained(hf_model_path_or_name)
        ori_model_class = MODEL_FOR_CAUSAL_LM_MAPPING[type(ori_hf_config)]
        ori_config_class = type(ori_hf_config)

        num_ori_layers = ori_hf_config.num_hidden_layers
        num_freezed_layers = len(model_a.config.expansion["freezed_layers"])
        assert num_exp_layers_a == num_ori_layers - num_freezed_layers, "Number of expanded layers from the config does not match the expected number of num_ori_layers - num_freezed_layers"
        split = max(int(num_ori_layers // num_exp_layers_a), 1)

        freezed_layers = list(model_a.config.expansion["freezed_layers"])
        weight_merger = WeightMerger(merge_method=merge_method_a)

        merged_layers = ModuleList()
        expanded_layers_a = model_a.model.layers
        expanded_layers_b = model_b.model.layers
        for layer_idx in range(num_ori_layers):
            merged_layers.append(expanded_layers_a[layer_idx])

            if (layer_idx + 1) % split == 0:
                merged_layer_a = expanded_layers_a[layer_idx]
                merged_layer_b = expanded_layers_b[layer_idx]

                if merge_method_a == "slerp":
                    assert isinstance(merged_layer_a, MergeLayerSlerp), f"Invalid merge layer type: {type(merged_layer_a)}, expected MergeLayerSlerp"
                elif merge_method_a == "lerp":
                    assert isinstance(merged_layer_a, MergeLayerLerp), f"Invalid merge layer type: {type(merged_layer_a)}, expected MergeLayerLerp"
                elif merge_method_a == "dlerp":
                    assert isinstance(merged_layer_a, MergeLayerDlerp), f"Invalid merge layer type: {type(merged_layer_a)}, expected MergeLayerDlerp"
                elif merge_method_a == "dlerpin":
                    assert isinstance(merged_layer_a, MergeLayerDlerpIn), f"Invalid merge layer type: {type(merged_layer_a)}, expected MergeLayerDlerpIn"
                else:
                    raise ValueError(f"Invalid merge method to merge weights: {merge_method_a}, supported: slerp, lerp")

                print(f"Model Expansion - merge back: Merging side car of the expanded layer at {layer_idx}")
                expanded_layer_a = merged_layer_a.expanded_layer
                expanded_layer_b = merged_layer_b.expanded_layer
                pretrained_layer = merged_layer_a.pretrained_layer

                # merge the expanded layer back to the pretrained layer by merging the weights
                interp_weight_a = torch.sigmoid(merged_layer_a.interpolation_factor)
                interp_weight_b = torch.sigmoid(merged_layer_b.interpolation_factor)
                print(f"Model Expansion - merge back: Layer: {layer_idx} Interpolation factor: model_a - {interp_weight_a} model_b - {interp_weight_b}")
                merged_layer = weight_merger.merge_weights_dual(pretrained_layer, expanded_layer_a, expanded_layer_b, interp_weight_a, interp_weight_b, merge_back=merge_back)

                # Replace the merged layer with the pretrained layer
                if merge_back:
                    merged_layers[-1] = merged_layer[0]  # merged_layer[0] is the merged pretrained_layer
                else:
                    merged_layer_a.expanded_layer = merged_layer[1]  # merged_layer[1] is the merged expanded_layer_a, this code is not necessary as it is updated in place, keep it for clarity
                    merged_layers[-1] = merged_layer_a
            else:
                assert layer_idx in freezed_layers, f"Model Expansion - merge back: Layer {layer_idx} should be freezed but not in freezed layers: {freezed_layers}"

        model_a.model.layers = merged_layers

        if merge_back:
            # update the config to original model config
            print(f"Model expansion - merge back: Updating model configuration with expand type: {expand_type_a} - {merge_method_b}, merged layers: {[i for i in range(num_ori_layers) if i not in freezed_layers]}")
            model_a.config.architectures = ori_hf_config.architectures
            assert model_a.config.num_hidden_layers == len(model_a.model.layers), f"Number of hidden layers in the model config ({model_a.config.num_hidden_layers}) does not match the number of layers in the model ({len(model_a.model.layers)})."
            model_a.config = ori_config_class.from_dict(model_a.config.to_dict())
            # still keep expansion_config: expand_type, expanded_from, merge_method, freezed_layers etc, for reference of being able to trace back on how the model was expanded and trained
            model_a.config.expansion = model_a.config.expansion
            model_a.config.expansion["num_exp_layers"] = -num_exp_layers_a  # change the num_exp_layers to negative to indicate the model is merged back

            # model is still the expanded model class, we need to update the model class to the original model class
            model_a.__class__ = ori_model_class

        return model_a


class MergeLayer(Module):
    """
    Base class for merging the output of two layers.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha=0, interpolation_loss_type='cosine', freeze_interpolation_factor=False, use_vllm=False):
        super().__init__()
        self.pretrained_layer = pretrained_layer
        self.expanded_layer = expanded_layer
        self.layer_idx = layer_idx
        self.interpolation_loss_alpha = interpolation_loss_alpha
        self.interpolation_loss_type = interpolation_loss_type
        self.freeze_interpolation_factor = freeze_interpolation_factor
        self.avg_interp_weight, self.num_interp_weight = 0.0, 0
        self.use_vllm = use_vllm

    def forward(self, *args, **kwargs):
        """
        Handle optional inputs such as past_key_value and do forward pass of side car layer.
        """
        # KV cache is global and updated in place, so we need a copy of instance to avoid update it twice in expanded_layer
        past_key_value = kwargs.pop('past_key_value', None)

        # If use_cache is False typically during training, past_key_value is None
        if past_key_value:
            # Inject custom logic to 'from transformers import Cache' to support the new model class of side car by monkey patching
            # This is to make sure the forward pass of the model works when use_cache is True(e.g. Inference with kv cache)
            # TODO: better make it one time injection, added here because this is the place knowing which cache class it is used when use_cache is True
            past_key_value.__getitem__ = cache____getitem__.__get__(past_key_value)
            past_key_value.get_seq_length = cache__get_seq_length.__get__(past_key_value)
            past_key_value.update = cache__update.__get__(past_key_value)

            # past_key_value.key_cache and past_key_value.value_cache should be expanded to tuple for MergeLayer
            # if key_cache and value cache are not added in current position self.layer_idx, add None to to indicate that cache at layer_idx is for MergeLayer, check code of model_expander.py->cache__update() on details
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(None)
                past_key_value.value_cache.append(None)

        idx = 1  # Start index for optional outputs
        output_attentions = kwargs.get('output_attentions', False)
        use_cache = kwargs.get('use_cache', False)

        if self.use_vllm:
            kv_cache = args[2]  # [4, num_blocks, block_size, num_kv_heads, head_size]
            hidden_states_input_pretrained, residual_input_pretrained = args[1], args[4]  # note that vLLM's DecoderLayer implementation updates the hidden_states and residual in place, which means after self.pretrained_layer, it is changed, so clone to prepare for forward pass of expanded_layer
            hidden_states_input_expanded, residual_input_expanded = args[1].clone(), args[4].clone() if args[4] is not None else None  # note that vLLM's DecoderLayer implementation updates the hidden_states and residual in place, which means after self.pretrained_layer, it is changed, so clone to prepare for forward pass of expanded_layer

        # Forward pass through the pretrained layer
        if past_key_value:
            output_pretrained = self.pretrained_layer(
                *args,
                past_key_value=past_key_value,
                **kwargs
            )
        else:  # for vllm to work
            if self.use_vllm:
                # Create a view for kv_cache_pretrained that references the first two elements of kv_cache
                kv_cache_pretrained = kv_cache[:2] if kv_cache is not None else None  # [2, num_blocks, block_size, num_kv_heads, head_size], first two are for pretrained_layer
                args = (args[0], hidden_states_input_pretrained, kv_cache_pretrained, args[3], residual_input_pretrained)  # replace the kv_cache in args, first one is for pretrained_layer

            output_pretrained = self.pretrained_layer(
                *args,
                **kwargs
            )

        # Forward pass through the expanded layer
        if past_key_value:
            output_expanded = self.expanded_layer(
                *args,
                past_key_value=past_key_value,
                **kwargs
            )
        else:
            if self.use_vllm:
                # Create a view for kv_cache_expanded that references the second two elements of kv_cache
                kv_cache_expanded = kv_cache[2:] if kv_cache is not None else None  # [2, num_blocks, block_size, num_kv_heads, head_size], first two are for pretrained_layer
                args = (args[0], hidden_states_input_expanded, kv_cache_expanded, args[3], residual_input_expanded)  # replace the kv_cache in args, first one is for pretrained_layer

            output_expanded = self.expanded_layer(
                *args,
                **kwargs
            )

        if self.use_vllm:
            # merge the residual and hidden states, this is special handling because vLLM delays the residual add operation to the layer norm forward pass compared to hugingface's transformers
            # however in our model expansion case, the merge operation may not be always linear, so the best strategy is to merge the hidden states and residual here and decouple it from layer norm.
            # TODO: this special handling may make vLLM slower/higher memory as it can't leverage the efficient fused_add_rms_norm operation anymore.
            # For linear merge such as LERP, it is more efficient to just add redisual in the end of this method, instead of return None for residual which is the current implementation to leverage vLLM's more efficent fused_add_rms_norm operation.
            # refer to: https://github.com/vllm-project/vllm/blob/eae3d48181b1ad27f132f14df18e8cff203f7552/vllm/model_executor/models/llama.py#L257
            # refer to: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/layernorm.py
            output_pretrained, output_expanded = self._add(output_pretrained), self._add(output_expanded)

        outputs = self.merge(output_pretrained, output_expanded, *args, **kwargs)

        # Handle optional outputs
        if output_attentions and len(output_pretrained) > idx and len(output_expanded) > idx:
            # Combine attentions using a tuple
            combined_attentions = (output_pretrained[idx], output_expanded[idx])
            outputs += (combined_attentions,)  # TODO: it might make sense to add (combined_attentions,) to be comptaible with some interpretability framework
            idx += 1

        if use_cache and len(output_pretrained) > idx and len(output_expanded) > idx:
            # Return present_key_value from output_expanded, present_key_value is updated already to fit in MergeLayer's tuple of kv cache design internally by both self.pretrained_layer() and self.expanded_layer()
            # Extract present_key_value and then add to outputs. Following the code convention of transformers' <ModelName>DecoderLayer(nn.Module).forward()
            present_key_value = output_expanded[idx]
            outputs += (present_key_value,)

        if self.use_vllm:
            # vLLM returns (hidden states, residual). For reidual, ideally we shall return None! as the residual is already added by self._add() method.
            # But for last layer, it needs to add residual to avoid error in vLLM's final Norm operation, refer to:
            # https://github.com/vllm-project/vllm/blob/eae3d48181b1ad27f132f14df18e8cff203f7552/vllm/model_executor/models/llama.py#L351
            # so the workaround here is to return residual with all zeros
            # residual = output_expanded[1]
            # residual.zero_()
            # outputs += (residual,)
            outputs += (None,)

        return outputs

    def _add(self, output):
        """
        Add the residual to the hidden states.
        """
        hidden_states, residual = output[0], output[1]
        # orig_dtype = hidden_states.dtype

        # # Convert to float32 to prevent numerical precision issue, follow vLLM implmenetation but may not be necessary as transformers' implementation works fine with original dtype
        # if hidden_states.dtype != torch.float32:
        #     hidden_states = hidden_states.to(torch.float32)
        # if residual.dtype != torch.float32:
        #     residual = residual.to(torch.float32)

        # Perform addition in-place to save memory. Only used for vLLM during inference: we don't need to worry about gradient computations.
        hidden_states.add_(residual)

        # # Convert back to original dtype
        # if hidden_states.dtype != orig_dtype:
        #     hidden_states = hidden_states.to(orig_dtype)
        # if residual.dtype != orig_dtype:
        #     residual = residual.to(orig_dtype)

        output = (hidden_states,) + (residual,) + output[2:]  # output_pretrained[2:]: in case vLLM adds more outputs in the future
        return output

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers.
        """
        raise NotImplementedError("Merge method not implemented")

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load weights into the model. This is for vllm to load the weights from the checkpoint.
        Refer to: https://docs.vllm.ai/en/v0.5.5/models/adding_model.html
        """
        for name, weight in weights:
            setattr(self, name, weight)

    def compute_divergence_loss(self, output_pretrained, output_expanded, interp_weight, *args, loss_type=None, margin=0.0, cosine='all_tokens', **kwargs):
        """
        Compute the divergence loss between the hidden states of the pretrained and expanded layers.

        Parameters:
        - output_pretrained: (hidden_states, attentions, present_key_value)
        - output_expanded: (hidden_states, attentions, present_key_value)
        - interp_weight: Scalar or tensor of shape (batch_size, seq_length, 1)
        - loss_type: 'cosine', 'dot', 'mse', or 'attention'
        - cosine: 'last_token', 'all_tokens', 'w_o_residual', used if loss_type is 'cosine' 

        Returns:
        - additional_loss: Scalar tensor representing the divergence loss.
        """
        # No need to compute divergence loss for vLLM inference to save time and GPU memory
        if self.use_vllm:
            return 0.0

        if loss_type is None:
            loss_type = self.interpolation_loss_type

        # this is for logging to tensorboard
        output_attentions = kwargs.get('output_attentions', False)
        hidden_states_pretrained = output_pretrained[0]  # output_pretrained[0] is already detached in MergeLayer->merge()
        hidden_states_expanded = output_expanded[0]
        cosine_similarity, magnitude_loss, attention_loss = None, None, None
        if output_attentions:
            attention_weights_pretrained, attention_weights_expanded = output_pretrained[1], output_expanded[1]
        else:
            attention_weights_pretrained, attention_weights_expanded = None, None

        # Determine the scaling factor alpha based on self.interpolation_loss_alpha
        if self.interpolation_loss_alpha == 0:
            # Log to TensorBoard for model explainability if needed
            self.log_to_tensorboard(hidden_states_pretrained, hidden_states_expanded, attention_weights_pretrained, attention_weights_expanded, cosine_similarity, magnitude_loss, attention_loss, alpha=interp_weight)
            return 0  # Divergence loss is disabled
        elif self.interpolation_loss_alpha == -1:
            alpha = interp_weight  # Dynamic scaling using interp_weight
        else:
            alpha = self.interpolation_loss_alpha  # Fixed scaling factor

        # This is for cosine similarity loss
        if self.use_vllm:  # vLLM decoder layer has input_hidden_state as the second argument
            input_hidden_state = args[1]  # Shape: (batch_size, seq_length, hidden_size)
        else:
            input_hidden_state = args[0]  # Shape: (batch_size, seq_length, hidden_size)

        def cosine_loss(hidden_states_pretrained, hidden_states_expanded, margin):
            if cosine == 'w_o_residual':
                # Hidden state w/o residual
                hidden_states_pretrained_net = hidden_states_pretrained - input_hidden_state  # (batch_size, seq_length, hidden_size)
                hidden_states_expanded_net = hidden_states_expanded - input_hidden_state  # (batch_size, seq_length, hidden_size)
            elif cosine == 'last_token':
                # Select only the last token's hidden state from each sequence in the batch, mask out the rest except the last one
                # bf16 is not considered a valid dtype for the condition tensor in torch.where()
                mask_last_token = torch.zeros_like(hidden_states_expanded, dtype=torch.bool)
                mask_last_token[:, -1, :] = 1  # Set the last token to 1

                # Prepare an identity vector
                identity_vector = torch.ones_like(hidden_states_pretrained) * 0.01  # Small constant vector across the dimension
                # Blend based on the mask
                hidden_states_pretrained_net = torch.where(mask_last_token, hidden_states_pretrained, identity_vector)
                hidden_states_expanded_net = torch.where(mask_last_token, hidden_states_expanded, identity_vector)
            elif cosine == 'all_tokens':
                # Select only the last token's hidden state from each sequence in the batch
                hidden_states_pretrained_net = hidden_states_pretrained
                hidden_states_expanded_net = hidden_states_expanded
            else:
                raise ValueError(f"Invalid cosine type: {cosine}, supported types: 'last_token', 'all_tokens', 'w_o_residual'")

            # Ensure tensors are float32 for precision
            hidden_states_pretrained_net = hidden_states_pretrained_net.float()  # (batch_size, seq_length, hidden_size)
            hidden_states_expanded_net = hidden_states_pretrained_net.float()  # (batch_size, seq_length, hidden_size)

            # Reshape tensors to (batch_size * seq_length, hidden_size) for computation
            hp = hidden_states_pretrained.view(-1, hidden_states_pretrained_net.size(-1))
            he = hidden_states_expanded.view(-1, hidden_states_expanded_net.size(-1))

            # Compute per-token cosine similarity
            cosine_similarity = F.cosine_similarity(hp, he, dim=-1, eps=1e-8)  # Shape: (batch_size * seq_length)
            # Compute the divergence based on the margin
            # Only penalize if cosine similarity is less than (1 - margin)
            # If cosine_similarity >= target_similarity, divergence is 0.
            # If cosine_similarity < target_similarity, divergence is positive and proportional to how much it falls below the target(margin).
            divergence = F.relu(1.0 - margin - cosine_similarity)  # Only positive divergences, # Shape: (batch_size * seq_length)

            # Compute the final additional loss. Element-wise multiplication and apply smooth L1 loss instead of L1!
            cosine_loss = F.smooth_l1_loss(divergence, torch.zeros(1, device=divergence.device, dtype=divergence.dtype), reduction='none')

            # Reshape back to (batch_size, seq_length)
            cosine_loss = cosine_loss.view(hidden_states_pretrained.size(0), -1)  # Shape: (batch_size, seq_length)

            return cosine_loss, hidden_states_pretrained_net, hidden_states_expanded_net

        if loss_type == 'cosine':
            # Compute cosine similarity between the hidden states along the last dimension (hidden_size) and Cosine similarity loss (1 - cosine similarity)
            # batch_size, seq_length, hidden_size = hidden_states_pretrained.shape
            # if self.target is None or self.target.size(0) != batch_size * seq_length:
            #     self.target = torch.ones(batch_size * seq_length, device=hidden_states_pretrained.device, dtype=hidden_states_pretrained.dtype)
            # divergence_loss = F.cosine_embedding_loss(hidden_states_pretrained.view(-1, hidden_size), hidden_states_expanded.view(-1, hidden_size), target=self.target, margin=0.0, reduction='none').view(batch_size, seq_length)  # Shape: (batch_size, seq_length)

            divergence_loss, _, _ = cosine_loss(hidden_states_pretrained, hidden_states_expanded, margin)

        elif loss_type == 'dot':
            # Compared to cosine, this captures both magnitude and alignment
            cosine_loss, hidden_states_pretrained_net, hidden_states_expanded_net = cosine_loss(hidden_states_pretrained, hidden_states_expanded, margin)

            # Compute magnitude difference loss
            hp_norm = torch.norm(hidden_states_pretrained_net, p=2, dim=-1)  # Shape: (batch_size, seq_length)
            he_norm = torch.norm(hidden_states_expanded_net, p=2, dim=-1)    # Shape: (batch_size, seq_length)
            magnitude_loss = (hp_norm - he_norm) ** 2  # Shape: (batch_size, seq_length)

            # Combine the losses, TODO: make it configurable
            beta = 0.5  # Weight for cosine similarity loss
            gamma = 0.5   # Weight for magnitude difference loss
            divergence_loss = beta * cosine_loss + gamma * magnitude_loss  # Shape: (batch_size, seq_length)

        # This is commented out because it caused the model to fail alignment during the experiment. The model quickly collapsed to essentially no intelligence in less than one epoch.
        # elif loss_type == 'dot':
        #     # We don't normalize here, so the dot product captures both magnitude and alignment

        #     # Compute the element-wise dot product between the hidden states (no normalization)
        #     dot_product = torch.sum(hidden_states_pretrained * hidden_states_expanded, dim=-1)

        #     # Compute the expected value when the vectors are identical (this is the squared magnitude of one of the vectors)
        #     # Since hidden_states_pretrained and hidden_states_expanded should be identical at minimum loss, we can use one of them.
        #     expected_dot_product = torch.sum(hidden_states_pretrained ** 2, dim=-1)

        #     # Normalize both the dot product and the expected value by a scaling factor depending on the hidden size.
        #     scaling_factor = hidden_states_pretrained.size(-1) ** 0.5  # Square root of hidden size
        #     dot_product = dot_product / scaling_factor
        #     expected_dot_product = expected_dot_product / scaling_factor

        #     # Compute the divergence loss as the squared difference between the dot product and the expected value
        #     divergence_loss = (dot_product - expected_dot_product) ** 2  # Shape: (batch_size, seq_length)

        elif loss_type == 'mse':
            # MSE loss between the hidden states of the pretrained and expanded layers
            hidden_states_pretrained = output_pretrained[0].float()
            hidden_states_expanded = output_expanded[0].float()

            # Compute the element-wise MSE loss without reduction, resulting shape: (batch_size, seq_length, hidden_size)
            mse_loss_fn = torch.nn.MSELoss(reduction='none')
            divergence_loss = mse_loss_fn(hidden_states_expanded, hidden_states_pretrained)

            # Reduce over the hidden_size dimension to match the shape from 'cosine' loss
            divergence_loss = divergence_loss.mean(dim=-1)  # Now shape: (batch_size, seq_length)

        elif loss_type == 'attention':
            # Assuming output_attentions=True, output_pretrained[1] and output_expanded[1] are attention weights
            if output_attentions or len(output_pretrained) <= 1 or len(output_expanded) <= 1:
                # attention_weights_pretrained and attention_weights_expanded should be of shape (batch_size, num_heads, seq_length, seq_length)
                # ensure attention weights are float32 for precision
                attention_weights_pretrained, attention_weights_expanded = output_pretrained[1].float(), output_expanded[1].float()

                # Detach the teacher's attention weights
                attention_weights_pretrained = attention_weights_pretrained
            else:
                raise ValueError("Attention weights are not available in the model outputs. Attention loss requires output_attentions=True in model forward pass.")

            # Compute log probabilities
            small_epsilon = 1e-8  # Small epsilon to prevent log(0)
            pre_attn_log = torch.log(attention_weights_pretrained + small_epsilon)  # Didn't add epsilon to prevent log(0), TODO: check if it's necessary
            exp_attn_log = torch.log(attention_weights_expanded + small_epsilon)

            # Compute element-wise KL divergence without reduction
            attention_loss = attention_weights_expanded * (exp_attn_log - pre_attn_log)  # Shape: (batch_size, num_heads, seq_length, seq_length)

            divergence_loss = attention_loss

            # Or alternatively mse_loss
            # # Compute the element-wise squared difference between attention maps
            # divergence_loss = F.mse_loss(attention_weights_expanded, attention_weights_pretrained, reduction='none')  # Shape: (batch_size, num_heads, seq_length, seq_length)

        else:
            raise ValueError(f"Invalid loss_type '{loss_type}'. Expected 'cosine', 'dot', 'mse', or 'attention'.")

        # Log to TensorBoard for model explainability
        self.log_to_tensorboard(hidden_states_pretrained, hidden_states_expanded, attention_weights_pretrained, attention_weights_expanded, cosine_similarity, magnitude_loss, attention_loss, alpha)

        # Adjust alpha to match divergence_loss shape for element-wise multiplication
        if isinstance(alpha, (float, int)) or (torch.is_tensor(alpha) and alpha.dim() == 0):
            # Alpha is a scalar; no adjustment needed
            pass
        else:
            assert torch.is_tensor(alpha), "Interpolation loss alpha must be a scalar or tensor."
            # Alpha is a tensor; adjust dimensions if necessary
            if alpha.dim() > divergence_loss.dim():  # Alpha shape: (batch_size, seq_length, 1) vs divergence_loss shape: (batch_size, seq_length)
                alpha = alpha.squeeze(-1)
            if divergence_loss.dim() == 4:  # Alpha shape: (batch_size, seq_length, 1) vs divergence_loss shape: (batch_size, num_heads, seq_length, seq_length)
                if alpha.dim() == 3:
                    alpha = alpha.unsqueeze(1)  # Expand interpolation_weight to (batch_size, 1, seq_length, 1)
                elif alpha.dim() == 2:
                    alpha = alpha.unsqueeze(1).unsqueeze(-1)  # Expand interpolation_weight to (batch_size, 1, seq_length, 1)

        # Compute the final additional loss. Element-wise multiplication and mean reduction
        additional_loss = (alpha * divergence_loss).mean()

        return additional_loss

    def log_to_tensorboard(self, hidden_states_pretrained, hidden_states_expanded, attention_weights_pretrained, attention_weights_expanded, cosine_similarity, magnitude_loss, attention_loss, alpha):
        """
        Log the additional loss to TensorBoard.
        """
        if not hasattr(self, 'model_logger') or self.model_logger is None or not hasattr(self, 'metadata') or self.metadata is None:
            return

        model_logger: ModelLogger = self.model_logger

        # self.metadata is dict of keys and values, e.g. {'labels': ['king-queen', "man-woman"], 'global_step': 0}
        # for each self.metadata{"labels"}, add layer_idx to the labels to differentiate the embeddings
        labels = [f"{label}-layer{self.layer_idx}" for label in self.metadata["labels"]]

        # Normalize the hidden states along the last dimension (hidden_size) and only for the last token
        hidden_states_pretrained_norm = F.normalize(hidden_states_pretrained[:, -1, :], p=2, dim=-1)  # Only log the hidden_states of the last token
        hidden_states_expanded_norm = F.normalize(hidden_states_expanded[:, -1, :], p=2, dim=-1)  # Shape: (batch_size, hidden_size)

        # Log embeddings to TensorBoard - separate graph for pretrained and expanded
        model_logger.add_embedding(hidden_states_pretrained_norm, metadata=labels, tag="Last token hidden state/Pretrained")
        model_logger.add_embedding(hidden_states_expanded_norm, metadata=labels, tag="Last token hidden state/Expanded")

        # Log embeddings to TensorBoard - one graph for pretrained and expanded
        model_logger.add_embedding(hidden_states_pretrained_norm, metadata=[f"[Pretrained]{label}" for label in labels], tag="Last token hidden state/Pretrained and Expanded")
        model_logger.add_embedding(hidden_states_expanded_norm, metadata=[f"[Expanded]{label}" for label in labels], tag="Last token hidden state/Pretrained and Expanded")

        # Calculate and log differences between consecutive embeddings
        for i in range(0, len(hidden_states_pretrained_norm) - 1, 2):  # Increment by 2 to handle pairs
            if i + 1 >= len(hidden_states_pretrained_norm):
                # Handle the case where the batch size is odd
                break

            pair_label_w_layer_idx = f'{labels[i]} to {labels[i+1]}'

            difference_pretrained = hidden_states_pretrained_norm[i + 1] - hidden_states_pretrained_norm[i]
            difference_expanded = hidden_states_expanded_norm[i + 1] - hidden_states_expanded_norm[i]
            # Log each difference as a separate embedding (treated as a point) to TensorBoard - separate graph for pretrained and expanded
            model_logger.add_embedding(difference_pretrained.unsqueeze(0), metadata=[pair_label_w_layer_idx], tag=f'Embedding Difference/Pretrained')
            model_logger.add_embedding(difference_expanded.unsqueeze(0), metadata=[pair_label_w_layer_idx], tag=f'Embedding Difference/Expanded')

            # Log each difference as a separate embedding (treated as a point) to TensorBoard - one graph for pretrained and expanded
            model_logger.add_embedding(difference_pretrained.unsqueeze(0), metadata=[f"[Pretrained]{pair_label_w_layer_idx}"], tag=f'Embedding Difference/Pretrained and Expanded')
            model_logger.add_embedding(difference_expanded.unsqueeze(0), metadata=[f"[Expanded]{pair_label_w_layer_idx}"], tag=f'Embedding Difference/Pretrained and Expanded')

            # Log the cosine similarity of the pair of embeddings to TensorBoard, note that it is already normalized so no need to F.cosine_similarity
            cosine_pretrained = torch.dot(hidden_states_pretrained_norm[i + 1], hidden_states_pretrained_norm[i])
            cosine_expanded = torch.dot(hidden_states_expanded_norm[i + 1], hidden_states_expanded_norm[i])
            pair_label_wo_layer_idx = f'{self.metadata["labels"][i]} to {self.metadata["labels"][i+1]}'
            model_logger.add_scalar(f'Layer{self.layer_idx}', cosine_pretrained, f'Cosine Similarity(PairLabel: One label vs Another)/Pretrained/{pair_label_wo_layer_idx}')
            model_logger.add_scalar(f'Layer{self.layer_idx}', cosine_expanded, f'Cosine Similarity(PairLabel: One label vs Another)/Expanded/{pair_label_wo_layer_idx}')

        if attention_weights_pretrained is not None and attention_weights_expanded is not None and attention_weights_pretrained.numel() > 0 and attention_weights_expanded.numel() > 0:  # Shape: (batch_size, num_heads, seq_length, seq_length)
            # Log attention weights to TensorBoard for side by side comparison, group by pair label(pair of the input prompt that are padded)
            model_logger.add_image(f'Attention Weights/Pretrained vs Expanded/{self.metadata["labels"]}', labels, attention_weights_pretrained, attention_weights_expanded)  # for the whole batch

            if attention_loss is None:  # Shape: (batch_size, num_heads, seq_length, seq_length)
                # Compute log probabilities
                small_epsilon = 1e-8  # Small epsilon to prevent log(0)
                pre_attn_log = torch.log(attention_weights_pretrained + small_epsilon)  # Didn't add epsilon to prevent log(0), TODO: check if it's necessary
                exp_attn_log = torch.log(attention_weights_expanded + small_epsilon)

                # Compute element-wise KL divergence without reduction
                attention_loss = attention_weights_expanded * (exp_attn_log - pre_attn_log)  # Shape: (batch_size, num_heads, seq_length, seq_length)
            model_logger.add_scalar(f'Layer{self.layer_idx}', attention_loss.mean(), f'Attention Difference(Label: Pretrained vs Expanded)/{self.metadata["labels"]}')  # for the whole batch

        if cosine_similarity is None:  # Shape: (batch_size, seq_length)
            cosine_similarity = F.cosine_similarity(hidden_states_pretrained, hidden_states_expanded, dim=-1, eps=0)
        model_logger.add_scalar(f'Layer{self.layer_idx}', cosine_similarity.mean(), f'Cosine Similarity(Label: Pretrained vs Expanded)/{self.metadata["labels"]}')  # for the whole batch

        if magnitude_loss is None:  # Shape: (batch_size, seq_length)
            # Compute magnitude difference loss
            hp_norm = torch.norm(hidden_states_pretrained, p=2, dim=-1)  # Shape: (batch_size, seq_length)
            he_norm = torch.norm(hidden_states_expanded, p=2, dim=-1)    # Shape: (batch_size, seq_length)
            magnitude_loss = (hp_norm - he_norm) ** 2  # Shape: (batch_size, seq_length)
        model_logger.add_scalar(f'Layer{self.layer_idx}', magnitude_loss.mean(), f'Magnitude Difference(Label: Pretrained vs Expanded)/{self.metadata["labels"]}')  # for the whole batch

        # Log the interpolation weight alpha to TensorBoard
        # Adjust alpha to match divergence_loss shape for element-wise multiplication
        if isinstance(alpha, (float, int)) or (torch.is_tensor(alpha) and alpha.dim() == 0):
            # Alpha is a scalar; no adjustment needed
            pass
        else:
            assert torch.is_tensor(alpha), "Interpolation loss alpha must be a scalar or tensor."
            alpha = alpha.mean()
        model_logger.add_scalar(f'Layer{self.layer_idx}', alpha, 'Interpolation Weight')


class MergeLayerProj(MergeLayer):
    """
    Projection Linear Interpolation Neurual Network merges the output of two layers by adding them together.
    Linear projection makes sure the output of the expanded layer is a linear transformation of the pretrained layer.
    Without the divergence loss, it is the same as MergeLayerLerp.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=True, use_vllm=False):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        if freeze_interpolation_factor:
            # Make it a frozen parameter to ensure it is part of self.named_parameters() and compatible with the weight-loading process
            self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype), requires_grad=False)  # Set requires_grad to False to freeze it
        else:
            # Make interpolation_factor a trainable parameter to adjust the interpolation factor
            self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype))

        # Both layers have the same output shape as we copied the weights from the original layer.
        # But it is designed to allow new/expanded layers to learn different representation constrainted by linear projection from pretrained, therefore we call this new linear projection layer

        # Initialize a projection layer to project the new layer to the same dimension as the pretrained layer
        self.linear_proj = Linear(expanded_layer.hidden_size, pretrained_layer.hidden_size, bias=False)

        # Initialize the weights of the projection layer to an identity matrix and biases to zero
        self._initialize_identity(self.linear_proj)

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers by adding them together after projecting the new layer to match the dimension of the pretrained
        """
        # Use sigmoid to constrain t between 0 and 1
        # This constraint ensures that t always represents a valid weighting between two outputs—preventing scenarios where t might lead to extrapolation rather than interpolation.
        interp_weight = torch.sigmoid(self.interpolation_factor)

        combined_output = torch.lerp(output_pretrained[0], output_expanded[0], interp_weight[0])

        # Compute divergence loss between hidden states and store it
        if self.interpolation_loss_alpha != 0 and self.use_vllm is False:
            # Project the pretrained layer by linear transformation and prepare for regularize the learning of hidden_states_expanded
            hidden_states_pretrained_proj = self.linear_proj(output_pretrained[0])
            output_pretrained = (hidden_states_pretrained_proj,) + output_pretrained[1:]

            # orthogonality_penalty = self.orthogonality_penalty(self.linear_proj.weight)

            # To ensure the expanded layer is a linear transformation of the pretrained layer, loss_type mse or cosine is recommended. TODO: orthogonality_penalty + mse does not show good results in the experiment, need to investigate more.
            # self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight[0], *args, **kwargs) + orthogonality_penalty
            self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight[0], *args, **kwargs)

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            current_mean = interp_weight.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        return (combined_output,)

    def orthogonality_penalty(self, A, scaling_factor=0.01):
        I = torch.eye(A.size(0), device=A.device)
        penalty = scaling_factor * torch.norm(torch.matmul(A, A.T) - I)
        return penalty

    def _initialize_identity(self, layer):
        """
        Initialize the weights of a Linear layer to an identity matrix and biases to zero.
        """
        if isinstance(layer, torch.nn.Linear):
            with torch.no_grad():
                # Ensure the weight matrix is square
                assert layer.weight.shape[0] == layer.weight.shape[1], "Weight matrix is not square, cannot initialize to identity."
                # Initialize weights to identity matrix
                hidden_size = layer.weight.shape[0]
                # Slightly perturbed identity matrix where small random values are added to the identity. Start off from being exactly identity, encouraging learning different transformations
                layer.weight.data.copy_(torch.eye(hidden_size, hidden_size) + 0.01 * torch.randn(hidden_size, hidden_size))
                # Initialize biases to zero
                if layer.bias is not None:
                    layer.bias.data.zero_()


class MergeLayerLerp(MergeLayer):
    """
    Merge the output of two layers by adding them together linearly
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=False, use_vllm=False):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        if interpolation_factor is not None:
            if freeze_interpolation_factor:
                # Make it a frozen parameter to ensure it is part of self.named_parameters() and compatible with the weight-loading process
                self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype), requires_grad=False)  # Set requires_grad to False to freeze it
            else:
                # Make interpolation_factor a trainable parameter to adjust the interpolation factor
                self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype))
        else:
            self.interpolation_factor = None

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers by adding them together linearly
        """
        # Use sigmoid to constrain t between 0 and 1
        # This constraint ensures that t always represents a valid weighting between two outputs—preventing scenarios where t might lead to extrapolation rather than interpolation.
        if self.interpolation_factor is not None:
            interp_weight = torch.sigmoid(self.interpolation_factor)
        else:
            interp_weight = (1,)
            # # detach output_pretrained to prevent gradients from flowing back to the pretrained model as it is only used for divergence loss
            # output_pretrained = (output_pretrained[0].detach(),) + output_pretrained[1:]

        # Combine the outputs. In this case, we simply add hiden states from both layers together and use torch.lerp for linear interpolation
        combined_output = torch.lerp(output_pretrained[0], output_expanded[0], interp_weight[0])

        # Compute divergence loss between hidden states and store it
        self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight[0], *args, **kwargs)

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            current_mean = interp_weight[0] if type(interp_weight) is tuple else interp_weight.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        return (combined_output,)


class MergeLayerSlerp(MergeLayer):
    """
    Merge the output of two layers using a spherical linear interpolation (SLERP)
    approach with an adjustable interpolation factor, approximating with linear
    interpolation when vectors are closely aligned.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=False, use_vllm=False, small_theta_threshold: float=1e-4):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        self.small_theta_threshold = small_theta_threshold
        if freeze_interpolation_factor:
            # Make it a frozen parameter to ensure it is part of self.named_parameters() and compatible with the weight-loading process
            self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype), requires_grad=False)  # Set requires_grad to False to freeze it
        else:
            # Make interpolation_factor a trainable parameter to adjust the interpolation factor
            self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype))

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers using spherical linear interpolation (SLERP)
        """
        # Detach hidden_states_pretrained to prevent gradients from flowing back, ensure float32 to maintain numerical precision
        dtype = output_pretrained[0].dtype
        # Ensure float32 to maintain numerical precision, otherwise cosine_similarity won't be 1 for identical hidden states
        hidden_states_pretrained = output_pretrained[0].float()
        hidden_states_expanded = output_expanded[0].float()

        # Normalize the outputs
        v1_norm = F.normalize(hidden_states_pretrained, p=2, dim=-1)
        v2_norm = F.normalize(hidden_states_expanded, p=2, dim=-1)

        # Compute cosine and safeguard against numerical instability
        cos_theta = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

        # Calculate theta and apply corrections for small values
        theta = torch.acos(cos_theta)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2) + 1e-8)
        sin_theta = torch.clamp(sin_theta, min=1e-6)

        # Use sigmoid to constrain t between 0 and 1
        # This constraint ensures that t always represents a valid weighting between two outputs—preventing scenarios where t might lead to extrapolation rather than interpolation.
        interp_weight = torch.sigmoid(self.interpolation_factor)

        # Handle cases where theta is very small
        output_attentions, use_cache = kwargs.get('output_attentions', False), kwargs.get('use_cache', False)
        if output_attentions is False and use_cache is False and len(output_expanded) > 1:
            # Hacky way of detecting if the model is vLLM, if so make it static graph. TODO: make it configurable
            # vLLM uses CUDA graph capturing, so we need to avoid conditionals based on tensor values inside the forward method, only for inference by vLLM
            combined_output = (1 - interp_weight[0]) * hidden_states_pretrained + interp_weight[0] *hidden_states_expanded
        else:
            if (theta < self.small_theta_threshold).any():  # Adjust threshold as needed
                combined_output = (1 - interp_weight[0]) * hidden_states_pretrained + interp_weight[0] *hidden_states_expanded
            else:
                s1 = torch.sin((1 - interp_weight[0]) * theta) / sin_theta
                s2 = torch.sin(interp_weight[0] * theta) / sin_theta

                combined_output = s1 * hidden_states_pretrained + s2 * hidden_states_expanded

        # Compute divergence loss between hidden states and store it
        self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight[0], *args, **kwargs)

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            current_mean = interp_weight.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        # Convert hidden_states back to original dtype
        combined_output = combined_output.to(dtype)

        return (combined_output,)


class MergeLayerDlerp(MergeLayer):
    """
    Merge the output of two layers using a dynamic linear interpolation (DLERP) based on features of both layers's output hidden states.
    approach with an adjustable interpolation factor.

    Args:
        pretrained_layer (nn.Module): The pretrained layer whose output will be merged.
        expanded_layer (nn.Module): The new or expanded layer whose output will be merged.
        layer_idx (int): The index of the layer in the model.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=False, use_vllm=False):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        class DynamicInterpWeight(Module):
            """Dynamic interpolation weight module with a trainable linear layer to learn feature dependent interpolation factor."""
            def __init__(self, input_features):
                super(DynamicInterpWeight, self).__init__()
                self.fc = Linear(input_features, 1)
                self.sigmoid = Sigmoid()

            def forward(self, x):
                return self.sigmoid(self.fc(x))

        input_features = pretrained_layer.hidden_size * 2
        self.interp_weight = DynamicInterpWeight(input_features)

        # Initializing weights and bias
        with torch.no_grad():
            # TODO: uncomment this zero initialization for further experiments
            # # Slightly perturbed zero weights where small random values are added for symmetry breaking
            # nn.init.zeros_(self.interp_weight.fc.weight)  # Initialize weights to zero
            # temp_weights = torch.empty_like(self.interp_weight.fc.weight)  # Add small perturbations using Kaiming Uniform initialization scaled down
            # nn.init.kaiming_uniform_(temp_weights, a=0, mode='fan_in', nonlinearity='linear')
            # self.interp_weight.fc.weight.add_(0.01 * temp_weights)

            # By setting the bias to 0 when interpolation_factor is configured as 0, ensure that the initial output of the sigmoid function of DynamicInterpWeight is exactly 0.5
            self.interp_weight.fc.bias.data.fill_(interpolation_factor)

        if freeze_interpolation_factor:
            # Freeze the bias term
            self.interp_weight.fc.bias.requires_grad = False

    # Special handling: let self.interpolation_factor refers to self.interp_weight.fc.bias to be consistent with other MergeLayer variants, used in WeightMerger
    @property
    def interpolation_factor(self):
        return self.interp_weight.fc.bias

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers using spherical linear interpolation (SLERP)
        """
        hidden_states_pretrained = output_pretrained[0]
        hidden_states_expanded = output_expanded[0]

        # Compute interpolation weights with features concat from both layers's output hidden states
        features = torch.cat([hidden_states_pretrained, hidden_states_expanded], dim=-1)
        interp_weight = self.interp_weight(features)  # shape (batch_size, seq_length, 1)
        one_minus_interp_weight = 1.0 - interp_weight

        # Compute linear interpolation
        combined_hidden_states = one_minus_interp_weight * hidden_states_pretrained + interp_weight * hidden_states_expanded

        # Compute divergence loss between hidden states and store it
        self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight, *args, **kwargs)

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            current_mean = interp_weight.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        return (combined_hidden_states,)


class MergeLayerDlerpIn(MergeLayer):
    """
    Merge the output of two layers using a dynamic linear interpolation (DLERPIN) based on features of the input hidden states
    approach with an adjustable interpolation factor.

    Args:
        pretrained_layer (nn.Module): The pretrained layer whose output will be merged.
        expanded_layer (nn.Module): The new or expanded layer whose output will be merged.
        layer_idx (int): The index of the layer in the model.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=False, use_vllm=False):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        class DynamicInterpWeight(Module):
            """Dynamic interpolation weight module with a trainable linear layer to learn feature dependent interpolation factor."""
            def __init__(self, input_features):
                super(DynamicInterpWeight, self).__init__()
                self.fc = Linear(input_features, 1)
                self.sigmoid = Sigmoid()

            def forward(self, x):
                return self.sigmoid(self.fc(x))

        input_features = pretrained_layer.hidden_size
        self.interp_weight = DynamicInterpWeight(input_features)

        # Initializing weights and bias
        with torch.no_grad():
            # TODO: uncomment this zero initialization for further experiments
            # # Slightly perturbed zero weights where small random values are added for symmetry breaking
            # nn.init.zeros_(self.interp_weight.fc.weight)  # Initialize weights to zero
            # temp_weights = torch.empty_like(self.interp_weight.fc.weight)  # Add small perturbations using Kaiming Uniform initialization scaled down
            # nn.init.kaiming_uniform_(temp_weights, a=0, mode='fan_in', nonlinearity='linear')
            # self.interp_weight.fc.weight.add_(0.01 * temp_weights)

            # By setting the bias to 0 when interpolation_factor is configured as 0, ensure that the initial output of the sigmoid function of DynamicInterpWeight is exactly 0.5
            self.interp_weight.fc.bias.data.fill_(interpolation_factor)

        if freeze_interpolation_factor:
            # Freeze the bias term
            self.interp_weight.fc.bias.requires_grad = False

    # Special handling: let self.interpolation_factor refers to self.interp_weight.fc.bias to be consistent with other MergeLayer variants, used in WeightMerger
    @property
    def interpolation_factor(self):
        return self.interp_weight.fc.bias

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers using spherical linear interpolation (SLERP)
        """
        if self.use_vllm:  # vLLM decoder layer has input_hidden_state as the second argument
            input_hidden_state = args[1]  # Shape: (batch_size, seq_length, hidden_size)
        else:
            input_hidden_state = args[0]  # Shape: (batch_size, seq_length, hidden_size)

        hidden_states_pretrained = output_pretrained[0]
        hidden_states_expanded = output_expanded[0]

        # Compute interpolation weights with features of the input hidden states, this is the only difference from MergeLayerDlerp!
        features = input_hidden_state
        interp_weight = self.interp_weight(features)  # shape (batch_size, seq_length, 1)
        one_minus_interp_weight = 1.0 - interp_weight

        # Compute linear interpolation
        combined_hidden_states = one_minus_interp_weight * hidden_states_pretrained + interp_weight * hidden_states_expanded

        # Compute divergence loss between hidden states and store it
        self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight, *args, **kwargs)

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            current_mean = interp_weight.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        return (combined_hidden_states,)


class MergeLayerMoE(MergeLayer):
    """
    Merge the output of two layers using a Mixture of Experts (MoE) approach with an adjustable interpolation factor.
    Note that we only consider two experts in this implementation.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=False, use_vllm=False):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        class MoEGating(nn.Module):
            def __init__(self, hidden_size):
                super(MoEGating, self).__init__()
                # Gating network to produce logits for the two experts
                self.gate_network = nn.Sequential(
                    # nn.Linear(hidden_size, hidden_size, dtype=parameter_tensor.dtype, device=parameter_tensor.device),  # TODO: uncomment to test with non-linear gating
                    # nn.ReLU(),
                    nn.Linear(hidden_size, 2)  # Explicit dtype and device assignment
                )
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, input_hidden_state):
                # Compute gate logits
                gate_logits = self.gate_network(input_hidden_state)  # Shape: (batch_size, seq_length, 2)
                # Apply softmax to get gating weights
                gate_weights = self.softmax(gate_logits)  # Shape: (batch_size, seq_length, 2)
                return gate_weights  # Shape: (batch_size, seq_length, 2)

        # Initialize the gating network
        hidden_size = pretrained_layer.hidden_size
        self.gating = MoEGating(hidden_size)

        # Initializing weights and bias
        with torch.no_grad():
            gate_output_layer = self.gating.gate_network[-1]

            # TODO: uncomment this zero initialization for further experiments
            # # Slightly perturbed zero weights where small random values are added for symmetry breaking
            # nn.init.zeros_(gate_output_layer.weight)  # Initialize weights to zero
            # temp_weights = torch.empty_like(gate_output_layer.weight)  # Add small perturbations using Kaiming Uniform initialization scaled down
            # nn.init.kaiming_uniform_(temp_weights, a=0, mode='fan_in', nonlinearity='linear')
            # gate_output_layer.weight.add_(0.01 * temp_weights)

            # Set biases to achieve desired initial gating weights
            # Biases are set to [0.0, interpolation_factor], this means, config interpolation_factor to negative value to favor pretrained layer, positive value to favor expanded layer
            gate_output_layer.bias.data.fill_(0.0)  # Set all biases to 0 initially
            gate_output_layer.bias.data[1] = interpolation_factor  # Set second bias to interpolation_factor

        if freeze_interpolation_factor:
            # Freeze the bias term corresponding to interpolation_factor
            gate_output_layer.bias.requires_grad = False

    @property
    def interpolation_factor(self):
        # Compute the inverse of the probability of favoring pretrained output to be compatible with other MergeLayer variants
        biases = self.gating.gate_network[-1].bias
        # Weight might be sharded, so we need to handle the case where bias is empty
        if self.gating.gate_network[-1].bias.shape == torch.Size([0]):
            return None
        probabilities = F.softmax(biases, dim=0)
        logit_output = torch.logit(probabilities[0])
        return logit_output

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        """
        Merge the output of two layers using a Mixture of Experts (MoE) approach.
        """
        if self.use_vllm:  # vLLM decoder layer has input_hidden_state as the second argument
            input_hidden_state = args[1]  # Shape: (batch_size, seq_length, hidden_size)
        else:
            input_hidden_state = args[0]  # Shape: (batch_size, seq_length, hidden_size)

        # Extract hidden states from the outputs
        hidden_states_pretrained = output_pretrained[0]  # Shape: (batch_size, seq_length, hidden_size)
        hidden_states_expanded = output_expanded[0]      # Shape: (batch_size, seq_length, hidden_size)

        # Compute the gating scores
        gate_scores = self.gating(input_hidden_state)  # Shape: (batch_size, seq_length, 2)

        # Select the top-1 expert for each token
        _, top_expert_indices = gate_scores.max(dim=-1)  # Shape: (batch_size, seq_length)

        # Combine the expert outputs by selecting the output from the selected expert
        combined_hidden_states = self._combine_expert_outputs(
            hidden_states_pretrained, hidden_states_expanded, top_expert_indices
        )  # Shape: (batch_size, seq_length, hidden_size)

        if self.interpolation_loss_alpha != 0 and self.use_vllm is False:
            # Compute divergence loss between hidden states and store it
            mask_expert = (top_expert_indices == 1).unsqueeze(-1)  # Extend mask for element-wise multiplication, shape: (batch_size, seq_length, 1)
            interp_weights = 1 - gate_scores[..., 0].unsqueeze(-1)  # gate_scores[..., 0] is the probability of selecting the pretrained expert, interp_weights is the probability of selecting the expanded expert

            # Prepare an identity vector
            identity_vector = torch.ones_like(hidden_states_pretrained) * 0.01  # Small constant vector across the dimension
            # Blend based on the mask
            hidden_states_pretrained_masked = torch.where(mask_expert, hidden_states_pretrained, identity_vector)
            hidden_states_expanded_masked = torch.where(mask_expert, hidden_states_expanded, identity_vector)
            output_pretrained = (hidden_states_pretrained_masked,) + output_pretrained[1:]
            output_expanded = (hidden_states_expanded_masked, ) + output_expanded[1:]

            # Compute divergence loss elements directly, zeroing out where not applicable
            # Note that although hidden_states_pretrained and hidden_states_expanded are masked, still mask the interp_weights because interp_weights is only used when self.interpolation_loss_alpha is -1
            self.additional_loss = self.compute_divergence_loss(
                output_pretrained,  # Identity vector where mask is False
                output_expanded,  # Identity vector where mask is False
                interp_weights * mask_expert,  # Apply weights only where mask is True, zero elsewhere
                *args,
                **kwargs
            )

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            interp_weights = gate_scores[..., 1].unsqueeze(-1)  # probability of selecting the expanded expert
            current_mean = interp_weights.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        return (combined_hidden_states,)

    def _combine_expert_outputs(self, hidden_states_pretrained, hidden_states_expanded, top_expert_indices):
        # Determine if we need to add a batch dimension
        unsqueezed = hidden_states_pretrained.dim() == 2
        if unsqueezed:
            hidden_states_pretrained = hidden_states_pretrained.unsqueeze(0)
            hidden_states_expanded = hidden_states_expanded.unsqueeze(0)
            top_expert_indices = top_expert_indices.unsqueeze(0)

        # Stack along the new expert dimension
        expert_outputs = torch.stack([hidden_states_pretrained, hidden_states_expanded], dim=2)

        # Prepare indices for gather: need to match the dimensions of expert_outputs
        indices = top_expert_indices.unsqueeze(-1).unsqueeze(-1)
        indices = indices.expand(-1, -1, 1, hidden_states_pretrained.size(-1))  # Shape: [batch_size, seq_length, 1, hidden_size]

        # Gather the outputs from the selected experts
        combined_hidden_states = torch.gather(expert_outputs, 2, indices).squeeze(2)

        # If we added a batch dimension for 2D inputs, remove it before returning
        if unsqueezed:
            combined_hidden_states = combined_hidden_states.squeeze(0)

        return combined_hidden_states


class MergeLayerProg(MergeLayer):
    """
    Progressive Linear Interpolation Neural Networks assume that the lateral connections help the expanded layer to access and utilize the pretrained features.
    """
    def __init__(self, pretrained_layer, expanded_layer, layer_idx, interpolation_factor=0.0, interpolation_loss_alpha=0, interpolation_loss_type="cosine", freeze_interpolation_factor=True, use_vllm=False):
        super().__init__(pretrained_layer, expanded_layer, layer_idx, interpolation_loss_alpha, interpolation_loss_type, freeze_interpolation_factor, use_vllm)

        if freeze_interpolation_factor:
            # Make it a frozen parameter to ensure it is part of self.named_parameters() and compatible with the weight-loading process
            self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype), requires_grad=False)  # Set requires_grad to False to freeze it
        else:
            # Make interpolation_factor a trainable parameter to adjust the interpolation factor
            self.interpolation_factor = Parameter(torch.full((1,), interpolation_factor, dtype=next(pretrained_layer.parameters()).dtype))

        # Assuming hidden_size is accessible from the pretrained layer
        hidden_size = pretrained_layer.hidden_size

        # Define the lateral connection as a Linear layer
        self.lateral_connection = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize the lateral connection weights to identity matrix
        self._initialize_identity(self.lateral_connection)

    def _initialize_identity(self, layer):
        """
        Initialize the weights of a Linear layer to an identity matrix and biases to zero.
        """
        if isinstance(layer, torch.nn.Linear):
            with torch.no_grad():
                # Ensure the weight matrix is square
                assert layer.weight.shape[0] == layer.weight.shape[1], "Weight matrix is not square, cannot initialize to identity."
                # Initialize weights to identity matrix
                layer.weight.data.copy_(torch.eye(layer.weight.shape[0]) + 0.01 * torch.randn(layer.weight.shape[0], layer.weight.shape[0]))
                # Initialize biases to zero
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def merge(self, output_pretrained, output_expanded, *args, **kwargs):
        # Extract hidden states from the outputs
        hidden_states_pretrained = output_pretrained[0]  # Shape: (batch_size, seq_length, hidden_size)
        hidden_states_expanded = output_expanded[0]      # Shape: (batch_size, seq_length, hidden_size)

        # Apply the lateral connection to the pretrained hidden states
        lateral_output_pretrained = self.lateral_connection(hidden_states_pretrained)

        # Combine the lateral output with the expanded layer's output
        interp_weight = torch.sigmoid(self.interpolation_factor)
        combined_output = (1 - interp_weight[0]) * lateral_output_pretrained + interp_weight[0] * hidden_states_expanded

        # Compute divergence loss between hidden states and store it
        self.additional_loss = self.compute_divergence_loss(output_pretrained, output_expanded, interp_weight[0], *args, **kwargs)

        # For debugging the model, report the running avg of interp_weight when model is in eval mode
        if not self.use_vllm and not self.training:
            current_mean = interp_weight.mean().item()
            self.num_interp_weight += 1
            self.avg_interp_weight += (current_mean - self.avg_interp_weight) / self.num_interp_weight

        return (combined_output,)


class ModelLogger:
    def __init__(self, global_step, tb_writer):
        self.emb_accumulator = {}
        self.labels_accumulator = {}
        self.scalar_accumulator = {}
        self.image_accumulator = {}
        self.global_step = global_step
        self.tb_writer = tb_writer

    def add_embedding(self, embeddings, metadata, tag):
        # Accumulate embeddings and labels
        self.emb_accumulator.setdefault(tag, []).extend(embeddings)
        self.labels_accumulator.setdefault(tag, []).extend(metadata)

    def add_scalar(self, tag, scalar, group):
        # Accumulate scalar values
        self.scalar_accumulator.setdefault(group, {}).setdefault(tag, []).append(scalar)

    def add_image(self, tag, labels, attention_weights_pretrained, attention_weights_expanded):
        self.image_accumulator.setdefault(tag, []).append((labels, attention_weights_pretrained.detach().cpu(), attention_weights_expanded.cpu()))

    def log_to_tensorboard(self):
        """
        Log accumulated embeddings, scalars, and images to TensorBoard.
        return: list of dict, each dict contains top_k keys with {'tag': tag, 'token_importance': token_importance, 'labels': labels}
        """
        # for each tag, concatenate all accumulated embeddings and labels
        for tag, embeddings in tqdm(self.emb_accumulator.items(), desc="Logging embeddings"):
            if embeddings[0].dim() == 1:
                embeddings_2d = [embedding.unsqueeze(0) for embedding in embeddings]            
            all_embeddings = torch.cat(embeddings_2d, dim=0)
            all_metadata = self.labels_accumulator[tag]
            self.tb_writer.add_embedding(all_embeddings, metadata=all_metadata, tag=tag, global_step=self.global_step)

        # for each group, log all scalar values at once, example of logging multiple scalars at once
        # tb_writer.add_scalars("Cosine Similarity - pair_label", {
        #     "Layer3": cosine3,
        #     "Layer7": cosine7,
        #     "Layer11": cosine11,
        #     "Layer15": cosine15
        # }, global_step)
        for group, scalars in tqdm(self.scalar_accumulator.items(), desc="Logging scalars"):
            scalars_dict = {tag: torch.mean(torch.cat([torch.as_tensor(s, dtype=torch.float32).unsqueeze(0) for s in scalar])) for tag, scalar in scalars.items()}
            self.tb_writer.add_scalars(group, scalars_dict, self.global_step)

        token_importance = []
        for tag, images in tqdm(self.image_accumulator.items(), desc="Logging attention weights"):
            labels, attention_weights_pretrained, attention_weights_expanded = zip(*images)

            # Flatten the labels list
            labels = [label for sublist in labels for label in sublist]

            # Concatenate the attention weights tensors along the batch dimension
            attention_weights_pretrained = torch.cat(attention_weights_pretrained, dim=0)
            attention_weights_expanded = torch.cat(attention_weights_expanded, dim=0)

            self._log_attention_matrix(tag, labels, attention_weights_pretrained, attention_weights_expanded)

            token_importance_per_tag = self._get_token_importance_attn_matrix(tag, labels, attention_weights_pretrained, attention_weights_expanded)
            token_importance.append(token_importance_per_tag)

        # Reset accumulators
        self.emb_accumulator = {}
        self.labels_accumulator = {}
        self.scalar_accumulator = {}
        self.image_accumulator = {}

        return token_importance

    def _log_attention_matrix(self, tag, labels, attention_weights_pretrained, attention_weights_expanded, head=-1):
        """Log attention weights to TensorBoard for side by side comparison"""

        # List to store all image tensors for batching
        image_tensors = []

        for i in range(0, len(attention_weights_pretrained) - 1, 2):  # Increment by 2 to handle pairs
            if i + 1 >= len(attention_weights_pretrained):
                # Handle the case where the batch size is odd
                break

            label_0, label_1 = labels[i], labels[i+1]

            # Extract attention matrices for the current batch and head, default to last head
            att_pretrained_0 = attention_weights_pretrained[i, head].to(torch.float32).numpy()
            att_pretrained_1 = attention_weights_pretrained[i + 1, head].to(torch.float32).numpy()
            # Plot side by side
            image_tensor = self._plot_side_by_side(att_pretrained_0, att_pretrained_1, f"[Pretrained]{label_0}", f"[Pretrained]{label_1}", head, self.global_step)
            image_tensors.append(image_tensor)

            att_expanded_0 = attention_weights_expanded[i, head].to(torch.float32).numpy()
            att_expanded_1 = attention_weights_expanded[i + 1, head].to(torch.float32).numpy()
            image_tensor = self._plot_side_by_side(att_expanded_0, att_expanded_1, f"[Expanded]{label_0}", f"[Expanded]{label_1}", head, self.global_step)
            image_tensors.append(image_tensor)

        # Stack all image tensors into a single tensor of shape (N, 3, H, W), N is the number of image pairs
        if image_tensors:
            images_batch = torch.stack(image_tensors)  # Shape: (N, 3, H, W)

            # Log the batch of images to TensorBoard
            self.tb_writer.add_images(tag, images_batch, global_step=0, dataformats='NCHW')

    def _plot_side_by_side(self, att_pretrained, att_expanded, label_0, label_1, head, step):
        """
        Plots two attention matrices side by side and returns the image.
        """
        import io
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot pretrained attention
        im0 = axes[0].imshow(att_pretrained, cmap='viridis')
        axes[0].set_title(f'{label_0} - Head {head} - Step {step}')
        fig.colorbar(im0, ax=axes[0])

        # Plot expanded attention
        im1 = axes[1].imshow(att_expanded, cmap='viridis')
        axes[1].set_title(f'{label_1} - Head {head} - Step {step}')
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')  # Ensure it's in RGB format
        image = np.array(image)

        plt.close(fig)

        # Convert to tensor and permute to (C, H, W)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        return image_tensor

    def _get_token_importance_attn_matrix(self, tag, labels, attention_weights_pretrained, attention_weights_expanded, head=None):
        # attention_weights_pretrained: shape (N, num_heads, seq_length, seq_length)
        # List to store mix of both attentions and labels from attention_weights_pretrained, attention_weights_expanded
        double_labels = []
        attentions = []

        for i in range(0, len(attention_weights_pretrained) - 1, 2):  # Increment by 2 to handle pairs
            if i + 1 >= len(attention_weights_pretrained):
                # Handle the case where the batch size is odd
                break

            label_0, label_1 = labels[i], labels[i+1]

            # Extract attention matrices for the current batch and head, default to last head
            att_pretrained_0 = attention_weights_pretrained[i, head] if head is not None else attention_weights_pretrained[i].mean(dim=0)
            attentions.append(att_pretrained_0)
            double_labels.append(label_0)
            att_pretrained_1 = attention_weights_pretrained[i + 1, head] if head is not None else attention_weights_pretrained[i + 1].mean(dim=0)
            attentions.append(att_pretrained_1)
            double_labels.append(label_1)

            att_expanded_0 = attention_weights_expanded[i, head] if head is not None else attention_weights_expanded[i].mean(dim=0)
            attentions.append(att_expanded_0)
            double_labels.append(label_0)
            att_expanded_1 = attention_weights_expanded[i + 1, head] if head is not None else attention_weights_expanded[i + 1].mean(dim=0)
            attentions.append(att_expanded_1)
            double_labels.append(label_1)

        # Stack all attention tensors into a single tensor of shape (2N, seq_length, seq_length)
        if attentions:
            attentions = torch.stack(attentions)
            token_importance = self._get_token_importance(attentions)  # Shape: (2N, K)
        else:
            token_importance = None

        return {'tag': tag, 'token_importance': token_importance, 'labels': double_labels}


    def _get_token_importance(self, attention_weights: torch.Tensor, alpha: float = 0.5, use_entropy_variance: bool = False) -> torch.Tensor:
        """
        Combine the last query's attention with global importance measures to determine top K keys.

        Args:
            attention_weights (torch.Tensor): Shape (N, num_queries, num_keys).
            alpha (float): Weighting factor to balance last query attention and global key importance.
            use_entropy_variance (bool): Whether to incorporate entropy and variance metrics.

        Returns:
            torch.Tensor: token_importance with shape (N, num_keys).
        """
        _, _, num_keys = attention_weights.size()  # Shape: (N, num_queries, num_keys)

        # Normalize attention weights
        attention_weights = F.softmax(attention_weights, dim=-1)  # Shape: (N, num_queries, num_keys)

        # Extract last query's attention
        last_query_attention = attention_weights[:, -1, :]  # Shape: (N, num_keys)

        if use_entropy_variance:
            # Compute entropy for each key across queries
            epsilon = 1e-12
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + epsilon), dim=1)  # Shape: (N, num_keys)
            entropy_normalized = entropy / torch.log(torch.tensor(num_keys, dtype=entropy.dtype, device=entropy.device))
            entropy_score = 1.0 - entropy_normalized  # Higher score for lower entropy

            # Compute variance for each key across queries
            key_variance = torch.var(attention_weights, dim=1)  # Shape: (N, num_keys)

            # Combine entropy and variance into a global importance score
            global_importance = 0.5 * entropy_score + 0.5 * key_variance  # Equal weighting; adjust as needed

            # Normalize global importance
            global_importance = F.normalize(global_importance, p=1, dim=1)  # Shape: (N, num_keys)

            # Combine last query attention with global importance
            combined_importance = alpha * last_query_attention + (1 - alpha) * global_importance  # Shape: (N, num_keys)
        else:
            # Simple hybrid: weighted sum of last query attention and global mean attention
            global_key_importance = attention_weights.mean(dim=1)  # Shape: (N, num_keys)
            combined_importance = alpha * last_query_attention + (1 - alpha) * global_key_importance  # Shape: (N, num_keys)

        # Normalize combined importance
        combined_token_importance = F.normalize(combined_importance, p=1, dim=1)  # Shape: (N, num_keys)

        return combined_token_importance


class WeightMerger:
    def __init__(self, merge_method='slerp'):
        """
        Initializes the merger with two models, the interpolation weight, and the method of interpolation.

        Args:
        merge_method (str): The method of interpolation; 'lerp' for Linear Interpolation or 'slerp' for Spherical Linear Interpolation.
        """
        self.merge_method = merge_method
        self.epsilon = 1e-6

    def merge_weights(self, pretrained_layer, expanded_layer, interp_weight=0.5):
        """
        Merges the weights of the pretrained layer towards those of the expanded layer using the specified interpolation method.

        Args:
            pretrained_layer (nn.Module): The first model whose parameters are to be updated.
            expanded_layer (nn.Module): The second model whose parameters will influence the first.
            interp_weight (float): The interpolation weight used for merging (0 to 1).
        """
        # Iterate over parameters in the pretrained layer
        for name, pretrained in pretrained_layer.named_parameters():
            # Ensure the parameter exists in the expanded layer
            if name in expanded_layer.state_dict():
                # Get the corresponding parameter from the expanded layer
                expanded_param = expanded_layer.state_dict()[name].data

                # Cast parameters to FP32 for higher precision computations
                pretrained_data_fp32 = pretrained.data.float()
                expanded_param_fp32 = expanded_param.float()

                if self.merge_method == 'lerp':
                    # Perform linear interpolation in FP32
                    merged = (1 - interp_weight) * pretrained_data_fp32 + interp_weight * expanded_param_fp32

                    # Update the parameter in BF16
                    with torch.no_grad():
                        pretrained.data.copy_(merged.to(pretrained.data.dtype))

                elif self.merge_method in ['slerp', 'dlerp', 'dlerpin']:
                    # Normalize the vectors in FP32
                    norm_pretrained_layer = torch.nn.functional.normalize(pretrained_data_fp32, p=2, dim=-1)
                    norm_expanded_layer = torch.nn.functional.normalize(expanded_param_fp32, p=2, dim=-1)

                    # Calculate the cosine and sine of the angle in FP32
                    cos_theta = torch.sum(norm_pretrained_layer * norm_expanded_layer, dim=-1, keepdim=True)
                    cos_theta = torch.clamp(cos_theta, -1.0 + self.epsilon, 1.0 - self.epsilon)
                    theta = torch.acos(cos_theta)
                    sin_theta = torch.sin(theta) + self.epsilon  # Add epsilon to avoid division by zero

                    # Perform spherical linear interpolation in FP32
                    s1 = torch.sin((1 - interp_weight) * theta) / sin_theta
                    s2 = torch.sin(interp_weight * theta) / sin_theta
                    merged = s1 * pretrained_data_fp32 + s2 * expanded_param_fp32

                    # Update the parameter in BF16
                    with torch.no_grad():
                        pretrained.data.copy_(merged.to(pretrained.data.dtype))
                else:
                    raise ValueError("Unsupported interpolation method specified.")

    def merge_weights_dual(self, pretrained_layer, expanded_layer_a, expanded_layer_b, interp_weight_a=0.5, interp_weight_b=0.5, merge_back=False) -> List[nn.Module]:
        """
        Merges the weights of the pretrained layer towards those of the expanded layer using the specified interpolation method.

        Args:
        pretrained_layer (nn.Module): The first model whose parameters are to be updated.
        expanded_layer_a (nn.Module): The first expanded model whose parameters will influence the first.
        expanded_layer_b (nn.Module): The second expanded model whose parameters will influence the first.
        interp_weight_a (float): The interpolation weight used for the first expanded model (0 to 1).
        interp_weight_b (float): The interpolation weight used for the second expanded model (0 to 1).
        merge_back (bool): Whether to merge the weights of expanded_layer_a and expanded_layer_b back to the pretrained_layer.

        return: merged layers
        """
        # Iterate over parameters in the pretrained layer
        for name, pretrained in pretrained_layer.named_parameters():
            # Ensure the parameter exists in the expanded layer
            if name in expanded_layer_a.state_dict():
                # Get the corresponding parameter from the expanded layer
                expanded_param_a = expanded_layer_a.state_dict()[name].data
                expanded_param_b = expanded_layer_b.state_dict()[name].data

                if self.merge_method == 'lerp':
                    # Perform linear interpolation and update the parameter in place
                    with torch.no_grad():
                        if merge_back:
                            interp_weight = (interp_weight_a + interp_weight_b) / 2
                            pretrained.data.copy_((1 - interp_weight) * pretrained.data + (interp_weight_a / 2) * expanded_param_a + (interp_weight_b / 2) * expanded_param_b)
                        else:
                            # merge expanded_layer_a and expanded_layer_b into expanded_layer_a
                            interp_weight_a = interp_weight_a / (interp_weight_a + interp_weight_b)
                            interp_weight_b = 1 - interp_weight_a
                            expanded_param_a.copy_(interp_weight_a * expanded_param_a + interp_weight_b * expanded_param_b)
                elif self.merge_method in ['slerp', 'dlerp', 'dlerpin']:
                    if merge_back:
                        # Normalize the vectors (for unit vector parameters)
                        norm_pretrained_layer = torch.nn.functional.normalize(pretrained.data, p=2, dim=-1)
                        norm_expanded_layer_a = torch.nn.functional.normalize(expanded_param_a, p=2, dim=-1)

                        # Calculate the cosine and sine of the angle for expanded_layer_a
                        cos_theta_a = torch.sum(norm_pretrained_layer * norm_expanded_layer_a, dim=-1, keepdim=True)
                        cos_theta_a = torch.clamp(cos_theta_a, -1.0 + self.epsilon, 1.0 - self.epsilon)
                        theta_a = torch.acos(cos_theta_a)
                        sin_theta_a = torch.sin(theta_a) + self.epsilon  # add epsilon to avoid division by zero

                        # Perform spherical linear interpolation
                        s1 = torch.sin((1 - interp_weight_a) * theta_a) / sin_theta_a
                        s2 = torch.sin(interp_weight_a * theta_a) / sin_theta_a
                        merged = s1 * pretrained + s2 * expanded_param_a

                        # Repeat the process for expanded_layer_b
                        norm_merged_layer = torch.nn.functional.normalize(merged.data, p=2, dim=-1)
                        norm_expanded_layer_b = torch.nn.functional.normalize(expanded_param_b, p=2, dim=-1)
                        cos_theta_b = torch.sum(norm_merged_layer * norm_expanded_layer_b, dim=-1, keepdim=True)
                        cos_theta_b = torch.clamp(cos_theta_b, -1.0 + self.epsilon, 1.0 - self.epsilon)
                        theta_b = torch.acos(cos_theta_b)
                        sin_theta_b = torch.sin(theta_b) + self.epsilon  # add epsilon to avoid division by zero
                        s1 = torch.sin((1 - interp_weight_b) * theta_b) / sin_theta_b
                        s2 = torch.sin(interp_weight_b * theta_b) / sin_theta_b
                        merged = s1 * merged + s2 * expanded_param_b

                        # Update the parameter
                        with torch.no_grad():
                            pretrained.data.copy_(merged)
                    else:  # Slerp merge expanded_layer_a and expanded_layer_b
                        # Normalize the vectors (for unit vector parameters)
                        norm_expanded_layer_a = torch.nn.functional.normalize(expanded_param_a, p=2, dim=-1)
                        norm_expanded_layer_b = torch.nn.functional.normalize(expanded_param_b, p=2, dim=-1)

                        # Calculate the cosine and sine of the angle for expanded_layer_a
                        cos_theta_a = torch.sum(norm_expanded_layer_a * norm_expanded_layer_b, dim=-1, keepdim=True)
                        cos_theta_a = torch.clamp(cos_theta_a, -1.0 + self.epsilon, 1.0 - self.epsilon)
                        theta_a = torch.acos(cos_theta_a)
                        sin_theta_a = torch.sin(theta_a) + self.epsilon

                        # Perform spherical linear interpolation
                        s1 = torch.sin((1 - interp_weight_a) * theta_a) / sin_theta_a
                        s2 = torch.sin(interp_weight_a * theta_a) / sin_theta_a
                        merged = s1 * expanded_param_a + s2 * expanded_param_b

                        # Update the parameter
                        with torch.no_grad():
                            expanded_param_a.copy_(merged)
                else:
                    raise ValueError("Unsupported interpolation method specified.")

        if merge_back:
            expanded_layer = None
        else:
            expanded_layer = expanded_layer_a
        return [pretrained_layer, expanded_layer]


class ModelComparer:
    def __init__(self, pretrained_model, expanded_model):
        """
        Accepts two models and compares their weights. Can be either path to the model or the model itself.
        """
        if isinstance(pretrained_model, str):
            self.pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model)
        else:
            self.pretrained_model = pretrained_model

        if isinstance(expanded_model, str):
            self.expanded_model = AutoModelForCausalLM.from_pretrained(expanded_model)
        else:
            self.expanded_model = expanded_model

        # Make sure the models are on the same cpu device
        self.pretrained_model_cpu = self.pretrained_model.__class__(self.pretrained_model.config)  # Create a new instance of the model
        self.pretrained_model_cpu.load_state_dict(self.pretrained_model.state_dict())

        # Create a copy of the expanded model on the CPU
        self.expanded_model_cpu = self.expanded_model.__class__(self.expanded_model.config)  # Create a new instance of the model
        self.expanded_model_cpu.load_state_dict(self.expanded_model.state_dict())

    def compare_model_weights(self) -> bool:
        """
        Compares the weights of the pretrained model with the expanded model.
        """
        # Compare weights of layers
        result = True
        for layer_idx, (pretrained_layer, expanded_layer) in enumerate(zip(self.pretrained_model_cpu.model.layers, self.expanded_model_cpu.model.layers)):
            # Compare the weights of the pretrained layer with the expanded layer
            if not self.compare_layer_weights(pretrained_layer, expanded_layer):
                result = False
                logging.info(f"Layer {layer_idx} does not match.")
            else:
                logging.info(f"Layer {layer_idx} matches.")

        # Compare rest of model weights
        # Get the set of all submodules in the pretrained model layers
        layer_submodules = set()
        for layer in self.pretrained_model.model.layers:
            layer_submodules.update(layer.modules())

        # Iterate over all modules in the expanded model and compare those not in layer_submodules
        for name, module in self.expanded_model_cpu.named_modules():
            # If the module has trainable parameters and is not part of the layer submodules
            if module not in layer_submodules and module != self.expanded_model_cpu:
                # Check if this module is a parent container of any layer submodule
                is_container_of_submodule = any(submodule in module.modules() for submodule in layer_submodules)

                if not is_container_of_submodule:
                    logging.info(f"Comparing none transformer layer - module: {name}")
                    for param_name, param in module.named_parameters():
                        if param_name in self.pretrained_model_cpu.state_dict():
                            pretrained_param = self.pretrained_model.state_dict()[param_name].data
                            if not torch.equal(param.data, pretrained_param):
                                result = False
                                logging.info(f"Parameter {param_name} in module {name} does not match.")
                            else:
                                logging.info(f"Parameter {param_name} in module {name} matches.")

        if result:
            logging.info("All weights match.")
        else:
            logging.info("Some weights do not match.")

        return result

    def compare_layer_weights(self, pretrained_layer, expanded_layer) -> bool:
        """
        Compare the weights of the pretrained layer with the expanded layer.
        """
        # Iterate over parameters in the pretrained layer
        for name, pretrained in pretrained_layer.named_parameters():
            # Ensure the parameter exists in the expanded layer
            if name in expanded_layer.state_dict():
                # Get the corresponding parameter from the expanded layer
                expanded_param = expanded_layer.state_dict()[name].data

                # Compare the weights
                if not torch.equal(pretrained.data, expanded_param):
                    return False

        return True


# Customized version of 'from transformers import Cache' to support side car expanded architecture
def cache____getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
    """
    Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
    sequence length.
    """
    if layer_idx < len(self):
        if isinstance(self.key_cache[layer_idx], tuple):
            return (self.key_cache[layer_idx][0], self.value_cache[layer_idx][0])
        else:
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
    else:
        raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")


# Customized version of 'from transformers import Cache' to support side car expanded architecture
def cache__get_seq_length(cache, layer_idx: Optional[int] = 0) -> int:
    """Returns the sequence length of the cached states. A layer index can be optionally passed."""
    # TODO: deprecate this function in favor of `cache_position`
    if len(cache.key_cache) <= layer_idx:
        return 0
    # if self.key_cache[layer_idx] is tuple, it means this layer of kv cache is side car expanded, and the first element is the original kv cache
    if isinstance(cache.key_cache[layer_idx], tuple):
        return cache.key_cache[layer_idx][0].shape[-2]
    else:
        return cache.key_cache[layer_idx].shape[-2]


# Customized version of 'from transformers import Cache' to support side car expanded architecture
def cache__update(
    cache,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    cache_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

    Parameters:
        key_states (`torch.Tensor`):
            The new key states to cache.
        value_states (`torch.Tensor`):
            The new value states to cache.
        layer_idx (`int`):
            The index of the layer to cache the states for.
        cache_kwargs (`Dict[str, Any]`, `optional`):
            Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

    Return:
        A tuple containing the updated key and value states.
    """
    # Update the number of seen tokens
    if layer_idx == 0:
        cache._seen_tokens += key_states.shape[-2]

    # Update the cache, cache.key_cache and cache.value_cache are either lists of tensors(for DecoderLayer) or list of tuples of tensors(for MergeLayer)
    if len(cache.key_cache) <= layer_idx:
        cache.key_cache.append(key_states)
        cache.value_cache.append(value_states)

        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        # if self.key_cache[layer_idx] is tuple or None, it means this position of kv cache is for MergeLayer
        # the first element is the kv cache of pretrained_layer and second element is the kv cache of expanded_layer in the MergeLayer
        if cache.key_cache[layer_idx] is None:  # None was appended to indicate that cache at layer_idx is for MergeLayer
            cache.key_cache[layer_idx] = (key_states,)
            cache.value_cache[layer_idx] = (value_states,)

            return cache.key_cache[layer_idx][0], cache.value_cache[layer_idx][0]
        elif isinstance(cache.key_cache[layer_idx], tuple):
            if len(cache.key_cache[layer_idx]) == 1:  # if second element of the cache is not updated yet, then update the second element. Note: don't change the calling sequence in MergeLayer->forward(), 1st: self.pretrained_layer() 2nd: self.expanded_layer()
                cache.key_cache[layer_idx] = (cache.key_cache[layer_idx][0], key_states)
                cache.value_cache[layer_idx] = (cache.value_cache[layer_idx][0], value_states)

                return cache.key_cache[layer_idx][1], cache.value_cache[layer_idx][1]
            else:  # len(cache.key_cache[layer_idx]) == 2
                if cache.key_cache[layer_idx][0].shape[-2] == cache.key_cache[layer_idx][1].shape[-2]:  # a hacky way of determining whether the cache is to be updated on the first or second element of the tuple, if first element is updated/concated already, then update the second element
                    cache.key_cache[layer_idx], cache.value_cache[layer_idx] = (torch.cat([cache.key_cache[layer_idx][0], key_states], dim=-2), cache.key_cache[layer_idx][1]), (torch.cat([cache.value_cache[layer_idx][0], value_states], dim=-2), cache.value_cache[layer_idx][1])

                    return cache.key_cache[layer_idx][0], cache.value_cache[layer_idx][0]
                else:
                    cache.key_cache[layer_idx], cache.value_cache[layer_idx] = (cache.key_cache[layer_idx][0], torch.cat([cache.key_cache[layer_idx][1], key_states], dim=-2)), (cache.value_cache[layer_idx][0], torch.cat([cache.value_cache[layer_idx][1], value_states], dim=-2))

                    return cache.key_cache[layer_idx][1], cache.value_cache[layer_idx][1]
        else:
            cache.key_cache[layer_idx] = torch.cat([cache.key_cache[layer_idx], key_states], dim=-2)
            cache.value_cache[layer_idx] = torch.cat([cache.value_cache[layer_idx], value_states], dim=-2)

            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


# Customized version of 'from vllm.model_executor.models.llama import LlamaModel' to support side car expanded architecture
from vllm.attention import AttentionMetadata
from vllm.sequence import IntermediateTensors
from vllm.distributed import get_pp_group
def llama_vllm__forward(
    self,
    input_ids: Optional[torch.Tensor],
    positions: torch.Tensor,
    kv_caches: List[torch.Tensor],
    attn_metadata: AttentionMetadata,
    intermediate_tensors: Optional[IntermediateTensors],
    inputs_embeds: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, IntermediateTensors]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,
            hidden_states,
            kv_caches[i - self.start_layer],
            attn_metadata,
            residual,
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors({
            "hidden_states": hidden_states,
            "residual": residual
        })

    # original code: hidden_states, _ = self.norm(hidden_states, residual)
    final_result = self.norm(hidden_states, residual)
    if isinstance(final_result, tuple):  # note that we return as is if the norm layer does not return a tuple, this is to support the case where last layer is MergeLayer which returns residual as None
       hidden_states, _ = self.norm(hidden_states, residual)
    else:
        hidden_states = final_result
    return hidden_states


# Customized version of 'from vllm.worker.cache_engine import CacheEngine' to support side car expanded architecture
from vllm.utils import is_pin_memory_available
def vllm_cache_engine___allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
    freezed_layers: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """Allocates KV cache on the specified device."""
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, self.block_size, self.num_kv_heads, self.head_size)
    # Expanded shape for side car expanded architecture
    # from: [2, num_blocks, block_size, num_kv_heads, head_size] to [4, num_blocks, block_size, num_kv_heads, head_size]
    kv_cache_shape_expanded = (kv_cache_shape[0]*2,) + kv_cache_shape[1:]
    pin_memory = is_pin_memory_available() if device == "cpu" else False
    kv_cache: List[torch.Tensor] = []
    for i in range(self.num_attention_layers):
        if freezed_layers is not None and i not in freezed_layers:
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape_expanded,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        else:
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
    return kv_cache


if __name__ == "__main__":
    import sys
    model_loading_config = sys.argv[1]  # e.g. '{"pretrained_model_name_or_path": "/shared/public/sharing/hawei/Meta-Llama-3-8B", "num_ori_layers": 48, "num_exp_layers": 48, "expand_type": "concat"}'
    exp_model_save_dir = sys.argv[2]  # e.g. /shared/public/sharing/model/llama-plus/seed-checkpoint-expand-8-stack
    resume_checkpoint_path = sys.argv[3]  # e.g. /shared/public/sharing/llama-plus/checkpoint-1000

    expander = ModelExpander(model_loading_config, resume_checkpoint_path)
    if expander.model:
        expander.expand_layers()
        expander.save_model(exp_model_save_dir)
    else:
        print(f"Model {expander.model_loading_config.pretrained_model_name_or_path} has already been expanded. Skipping expansion.")
