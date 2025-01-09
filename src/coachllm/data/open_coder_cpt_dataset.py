import logging
from pathlib import Path
from typing import Optional

from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from controlllm.configs.datasets import AbstractDataset as DatasetConfig
from controlllm.data.utils import sample_dataset


class OpenCoderCPTAnnealingDataset(Dataset):
    # list of important attributes: hf_dataset (HFDataset), tokenizer (PreTrainedTokenizer), dataset_config (DatasetConfig)
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        split: str,
    ):
        self.dataset_config = dataset_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        # take the split of dataset as the cache folder
        self.dataset_map_cache_dir = Path(self.dataset_config.dataset_map_cache_dir) / (split if not isinstance(split, list) else "__".join(split))

        # create the folder is it doesn't exist
        def ensure_dir_exists(dir_path):
            dir_path.resolve().mkdir(parents=True, exist_ok=True)

        ensure_dir_exists(self.dataset_map_cache_dir)
        try:
            logging.info(f"Loading data from path: {split}")
            # the dataset is not big enough to be necessary to load with streaming
            all_datasets = []
            for name in dataset_config.names:
                subset = load_from_disk(Path(dataset_config.hf_hub_dataset_cache_dir) / dataset_config.dataset.replace("/", "___") / name / split)

                logging.info(f"Sampling the dataset: {dataset_config.dataset} - {split}")
                subset = sample_dataset(subset, dataset_config, split)
                logging.info(f"Finished sampling the dataset: {dataset_config.dataset} - {split}")

                all_datasets.append(subset)

            self.hf_dataset = concatenate_datasets(all_datasets)

            # check if the dataset is loaded
            if self.hf_dataset is None:
                raise ValueError(f"No data loaded from path: {split}")
            logging.info(f"Finished loading data from path: {split}")
            logging.info(f"Sampling the dataset: {self.dataset_config.dataset} - {split}")
            self.hf_dataset = sample_dataset(self.hf_dataset, self.dataset_config, split)
            logging.info(f"Finished sampling the dataset: {self.dataset_config.dataset} - {split}")
            logging.info(f"Converting the dataset to features: {self.dataset_config.dataset}")
            self.convert_to_features()
            logging.info(f"Finished converting the dataset to features: {self.dataset_config.dataset}")
        except Exception as e:
            logging.exception(f"Loading of job dataset failed!: {e}")
            raise

    def __len__(self):
        return self.hf_dataset.shape[0]

    def convert_to_features(self):
        def tokenize_add_label(sample):
            # don't do padding here, it will be done in the dataloader. don't need to do truncation if packing is true
            prompt_ids = self.tokenizer.encode(self.tokenizer.bos_token + sample[self.dataset_config.prompt_columns],
                                               truncation=self.dataset_config.truncation,
                                               max_length=self.dataset_config.max_length,
                                               add_special_tokens=False)

            label_ids = self.tokenizer.encode("" if self.dataset_config.prompt_columns == self.dataset_config.response_column else self.dataset_config.response_column + self.tokenizer.eos_token,
                                              truncation=self.dataset_config.truncation,
                                              max_length=self.dataset_config.max_length,
                                              add_special_tokens=False)

            sample = {
                "input_ids": (prompt_ids + label_ids),
                "attention_mask": [1] * len(prompt_ids + label_ids),
                # apply -100 in prompt if instruction fine tuning else compute the loss for all tokens
                "labels": (prompt_ids + label_ids) if self.dataset_config.pretrain else ([-100] * len(prompt_ids) + label_ids),
            }
            return sample
        # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
        per_tokz_cache_folder = Path(self.tokenizer.name_or_path).name if Path(self.tokenizer.name_or_path).exists() else self.tokenizer.name_or_path
        per_tokz_cache_dir = self.dataset_map_cache_dir / per_tokz_cache_folder
        logging.info(f"Transforming the dataset with tokenization {per_tokz_cache_folder}: {self.dataset_config.dataset}")
        # create the folder is it doesn't exist
        if not per_tokz_cache_dir.exists():
            per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
        self.hf_dataset = self.hf_dataset.map(tokenize_add_label,
                                              num_proc=self.dataset_config.max_workers,
                                              cache_file_name=str(per_tokz_cache_dir / "tokenize_add_label.arrow"),
                                              load_from_cache_file=False if self.dataset_config.force_refresh else True,
                                              remove_columns=list(self.hf_dataset.features)
                                              )

    def __getitem__(self, index):
        return self.hf_dataset[int(index)]


# added this to allow testing with mock data
# !!! don't forget to map the dataset to its get function such as this one in DATASET_PREPROC of /data/__init__.py !!!
def get_dataset(
    dataset_config: DatasetConfig, tokenizer: PreTrainedTokenizer, split: Optional[str] = None
) -> Dataset:
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if split is None or dataset_config.local_test:
        # this is for local testing
        currPath = Path.cwd() / "controlllm" / "data" / "mock_data" / "job"
        logging.info(f"Loading dataset {currPath}")
        split = str(currPath)
    dataset = OpenCoderCPTAnnealingDataset(
        dataset_config,
        tokenizer=tokenizer,
        split=split
    )

    return dataset.hf_dataset
