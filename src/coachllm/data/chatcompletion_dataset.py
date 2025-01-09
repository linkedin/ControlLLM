import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from controlllm.configs.datasets import AbstractDataset as DatasetConfig
from controlllm.data.utils import load_avro_dirs_as_dataset, tokenize_dialog, sample_dataset


class ChatCompletion(Dataset):
    # list of important attributes: hf_dataset (HFDataset), tokenizer (PreTrainedTokenizer), dataset_config (DatasetConfig)
    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
    ):
        self.dataset_config = dataset_config
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.print_text = dataset_config.print_text
        self.data_path = data_path
        # take the split of dataset as the cache folder
        per_dp_cache_folder = Path(self.data_path).name
        self.dataset_map_cache_dir = Path(self.dataset_config.dataset_map_cache_dir) / per_dp_cache_folder
        self.dataset_raw_cache_dir = Path(self.dataset_config.avro_cache_dir) / per_dp_cache_folder

        # create the folder is it doesn't exist
        def ensure_dir_exists(dir_path):
            dir_path.resolve().mkdir(parents=True, exist_ok=True)

        ensure_dir_exists(self.dataset_map_cache_dir)
        ensure_dir_exists(self.dataset_raw_cache_dir)
        try:
            logging.info(f"Loading data from path: {data_path}")
            # the dataset is not big enough to be necessary to load with streaming
            self.hf_dataset: HFDataset = load_avro_dirs_as_dataset(
                avro_dir=self.data_path, avro_dataset_cache_dir=self.dataset_raw_cache_dir, max_workers=self.dataset_config.max_workers)
            # check if the dataset is loaded
            if self.hf_dataset is None:
                raise ValueError(f"No data loaded from path: {self.data_path}")
            logging.info(f"Finished loading data from path: {self.data_path}")
            logging.info(f"Sampling the dataset: {self.dataset_config.dataset} - {self.data_path}")
            self.hf_dataset = sample_dataset(self.hf_dataset, self.dataset_config, data_path)
            logging.info(f"Finished sampling the dataset: {self.dataset_config.dataset} - {self.data_path}")            
            logging.info(f"Converting the dataset to features: {self.dataset_config.dataset}")
            self.convert_to_features()
            logging.info(f"Finished converting the dataset to features: {self.dataset_config.dataset}")
        except Exception as e:
            logging.exception(f"Loading of dataset {self.data_path} failed!: {e}")
            raise

    def __len__(self):
        return self.hf_dataset.shape[0]

    def convert_to_features(self):
        # tokenize the dialog
        def tokenize_add_label(sample):
            return tokenize_dialog(dialog=sample[self.dataset_config.prompt_columns], tokenizer=self.tokenizer, dataset_config=self.dataset_config)

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
    dataset_config: DatasetConfig, tokenizer: PreTrainedTokenizer, data_path: Optional[str] = None
) -> Dataset:
    """cover function for handling loading the working dataset"""
    """dataset loading"""
    if data_path is None or dataset_config.local_test:
        # this is for local testing
        currPath = Path.cwd() / "controlllm" / "data" / "mock_data" / "job"
        logging.info(f"Loading dataset {currPath}")
        data_path = str(currPath)
    dataset = ChatCompletion(
        dataset_config,
        tokenizer=tokenizer,
        data_path=data_path
    )

    return dataset.hf_dataset
