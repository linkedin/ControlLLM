import logging
import inspect
from typing import List, Any, Optional, Tuple
from dataclasses import field, fields, make_dataclass, asdict, MISSING
from peft import LoraConfig as PeftLoraConfig, AdaptionPromptConfig, PrefixTuningConfig, PeftConfig

from controlllm.configs import LoraConfig, PrefixConfig, LlamaAdapterConfig
from controlllm.configs import TrainConfig, TrainConfigCommon, DatasetConfig, TokenizerLoadingConfig, ModelLoadingConfig, SetupConfig, FsdpConfig, WandbConfig
from controlllm.configs.datasets import AbstractDataset


class Configs:
    # Define the configuration parameters
    train_config: TrainConfigCommon = TrainConfig()
    # one of configs.datasets.py will be used based on train_config.dataset
    dataset_configs: List[AbstractDataset] = None  # class variable as placeholder, not in use
    model_loading_config = ModelLoadingConfig()
    tokenizer_loading_config = TokenizerLoadingConfig()
    setup_config = SetupConfig()
    fsdp_config = FsdpConfig()
    wandb_config = WandbConfig()

    # one of these will be used based on train_config.peft_method
    lora_config = LoraConfig()
    prefix_config = PrefixConfig()
    llama_adapter_config = LlamaAdapterConfig()
    # peft_config: 1 of (PeftLoraConfig, AdaptionPromptConfig, PrefixTuningConfig) based on train_config.peft_method
    peft_config: Optional[PeftConfig] = None  # class variable as placeholder, not in use

    def __init__(self, **kwargs):
        # Get all the config dataclasses
        configs = [getattr(Configs, attr) for attr in dir(Configs) if attr.endswith("_config") and getattr(Configs, attr)]
        # log default parameters
        logging.info("Default parameters:")
        self._log_configs(configs)

        # update config with command line arguments
        self._update_cfg(configs, **kwargs)
        # log updated parameters from command line
        logging.info("Updated parameters from command line:")
        self._log_configs(configs)

        # generate dataset config into self.train_config.dataset
        self.generate_dataset_cfg(**kwargs)
        configs.extend(self.dataset_configs)
        # log updated parameters from command line
        logging.info("Updated parameters from command line:")
        self._log_configs(self.dataset_configs)

        # correct the parameters
        self.model_loading_config.__post_init__(self.train_config)
        # log updated parameters from command line
        logging.info("Updated parameters from correction:")
        self._log_configs(self.model_loading_config)

        # generate PEFT config into self.peft_config
        if self.train_config.use_peft:
            self.generate_peft_cfg(**kwargs)
            configs.append(self.peft_config)
            # log updated parameters from command line
            logging.info("Updated parameters from correction:")
            self._log_configs(self.peft_config)

        # create a new dataclass for all configs to check for duplication
        self.args = self._combine_dataclasses_flat(configs)()
        logging.info(f"All the params that can be customized with command line and their default value: {asdict(self.args)}")

    # Combine all the arguments into flat args
    def _combine_dataclasses_flat(self, instances: List[Any]) -> Any:
        field_list: List[Tuple[str, Any, Any]] = []
        for instance in instances:
            class_name = instance.__class__.__name__
            for attributes in fields(instance):
                # Append tuple of (field_name, field_type, field_default_value)
                default_value = field(default_factory=attributes.default_factory,) \
                    if attributes.default_factory is not None and attributes.default_factory != MISSING else attributes.default
                # avoid duplicate names in the combined dataclass with class_name prefix
                field_list.append((class_name + "__" + attributes.name, attributes.type, default_value))

        # Create a new dataclass with combined fields
        return make_dataclass("CombinedArguments", field_list)

    # log the configuration parameters
    def _log_configs(self, configs: Any) -> None:
        if isinstance(configs, (tuple, list)):
            for c in configs:
                self._log_configs(c)
            return
        if configs is None:
            return
        config_name = type(configs).__name__
        config_params = asdict(configs)  # Convert dataclass instance to dict
        logging.info(f"{config_name} params: {config_params}")

        # log the parameters that are not converted by asdict (e.g. parameters not typed)
        # first collect them into dict and the log them
        params = {}
        for attr in dir(configs):
            if not attr.startswith('__') and not callable(getattr(configs, attr)):
                if attr not in config_params:
                    params[attr] = getattr(configs, attr)
        if params:
            logging.info(f"{config_name} params - remaining: {params}")

    # generate dataset confg based on train_config.dataset
    def generate_dataset_cfg(self, **kwargs):
        names = tuple({k: v for k, v in inspect.getmembers(DatasetConfig)}.keys())
        self.dataset_configs = []
        for dataset in self.train_config.dataset:
            assert dataset in names, f"Unknown dataset: {dataset}"

            # generate dataset_configs dynamically by train_config.dataset, all supported datasets are in configs/datasets.py
            dataset_configs = {k: v for k, v in inspect.getmembers(DatasetConfig)}[dataset]()
            # TODO: allow kwargs to specify per dataset config by prefixing dataset name
            self._update_cfg(dataset_configs, **kwargs)
            self.dataset_configs.append(dataset_configs)

    # generate peft config based on train_config.peft_method
    def generate_peft_cfg(self, **kwargs):
        configs = (self.lora_config, self.llama_adapter_config, self.prefix_config)
        peft_configs = (PeftLoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
        names = tuple(c.__name__.rstrip("_config") for c in configs)

        assert self.train_config.peft_method in names, f"Peft config not found: {self.train_config.peft_method}"

        config = configs[names.index(self.train_config.peft_method)]

        params = asdict(config)
        self.peft_config = peft_configs[names.index(self.train_config.peft_method)](**params)
        self._update_cfg(self.peft_config, **kwargs)

    def _update_cfg(self, config, **kwargs):
        if isinstance(config, (tuple, list)):
            for c in config:
                self._update_cfg(c, **kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                elif "." in k:
                    # allow --some_config.some_param=True
                    config_name, param_name = k.split(".")
                    if type(config).__name__ == config_name:
                        if hasattr(config, param_name):
                            setattr(config, param_name, v)
                        else:
                            # In case of specialized config we can warn user
                            print(f"Warning: {config_name} does not accept parameter: {k}")
                elif isinstance(config, TrainConfig):
                    print(f"Warning: unknown parameter {k}")
