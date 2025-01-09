import torch
import os
import copy
import logging
from pathlib import Path
from functools import partial
from typing import Any, Dict, Optional, List
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset, concatenate_datasets, DatasetDict

from transformers import default_data_collator
from transformers.data import DataCollatorForSeq2Seq

from controlllm.utils.config_utils import Configs
from controlllm.utils.dataset_sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler, CustomDistributedSampler, create_batches

from controlllm.data import DATASET_PREPROC
from controlllm.configs.datasets import AbstractDataset
from controlllm.configs.training import TrainConfigCommon


class DataLoaderWrapper:
    # list of important attributes: dataset_train (torch.utils.data.Dataset), dataset_val (torch.utils.data.Dataset),
    # train_dataloader (torch.utils.data.DataLoader), eval_dataloader (torch.utils.data.DataLoader), configs, tokenizer
    def __init__(self, configs: Configs, tokenizer):
        self.configs = configs
        self.tokenizer = tokenizer

        # prepare train_dataloader and eval_dataloader
        self.prepare_data_loader()

    def prepare_data_loader(self):
        # Load and preprocess the dataset for training and validation generically based on dataset_confg
        self.dataset_train = []
        self.dataset_val = []
        for dataset_config in self.configs.dataset_configs:
            try:
                dataset_train: BatchDataset = self.get_preprocessed_dataset(dataset_config, split="train")
                logging.info(f"--> Training Set {dataset_train.dataset_config.dataset} Length = {len(dataset_train)} Token count = {dataset_train.num_tokens}. Max token length = {max(dataset_train.lengths) if hasattr(dataset_train, 'lengths') and dataset_train.lengths else 'N/A'}. Min token length = {min(dataset_train.lengths) if hasattr(dataset_train, 'lengths') and dataset_train.lengths else 'N/A'}")
                logging.info(f"--> Training Set example: {self.tokenizer.decode(dataset_train[0]['input_ids'])}")

                # Note that dataset_config.run_validation enables dataset level control to be included in evaluation or not
                if (self.configs.train_config.run_validation or dataset_config.include_val) and dataset_config.run_validation:
                    if dataset_config.train_split == dataset_config.test_split:  # some datasets does not have test split, so use train split for validation if required
                        if dataset_config.test_split_ratio == -1:  # Special: set split_ratio to -1 to use the full dataset for training and testing which makes sense in pretraining
                            logging.info(f"Set validation set the same as training set for {dataset_config.dataset}. {dataset_config.train_split=} is the same as {dataset_config.test_split=}")
                            if dataset_config.mixed_training and dataset_config.pretrain is False:  # deep copy is to separate the training and testing set preparing for special handling of mixed training
                                logging.info(f"Deep copying the training set for validation set for {dataset_config.dataset} to prepare for mixed training...")
                                dataset_val = BatchDataset(self.configs.train_config, dataset_config, hf_dataset=copy.deepcopy(dataset_train.hf_dataset), load_from_disk=True, split="Test")
                            else:
                                dataset_val = dataset_train
                        else:  # split the dataset_train into train and test
                            logging.info(f"Splitting the training set for validation set for {dataset_config.dataset} with test_size={dataset_config.test_split_ratio}...")
                            dataset_train, dataset_val = dataset_train.train_test_split(test_size=dataset_config.test_split_ratio, seed=self.configs.setup_config.random_seed)
                    else:
                        dataset_val = self.get_preprocessed_dataset(dataset_config, split="test")

                    logging.info(f"--> Validation Set {dataset_val.dataset_config.dataset} Length = {len(dataset_val)} Token count = {dataset_val.num_tokens}. Max token length = {max(dataset_val.lengths) if hasattr(dataset_val, 'lengths') and dataset_val.lengths else 'N/A'}. Min token length = {min(dataset_val.lengths) if hasattr(dataset_val, 'lengths') and dataset_val.lengths else 'N/A'}")
                    logging.info(f"--> Validation Set example: {self.tokenizer.decode(dataset_train[0]['input_ids'])}")
                    self.dataset_val.append(dataset_val)

                # if dataset_config.include_val is True, include validation set in training, this only makes sense in pretraining
                if dataset_config.include_val:
                    logging.info(f"Training Set {dataset_train.dataset_config.dataset}: include validation set because {dataset_config.include_val}, adding additional length = {len(dataset_val)} to training set...")
                    train_datasets = [dataset_train.hf_dataset, dataset_val.hf_dataset]
                    dataset_train.hf_dataset = concatenate_datasets(train_datasets)

                # Special handling for mixed training: set labels to be the same as input_ids for pretraining on SFT data.
                # Mixed training involves using a combination of Pretrain and SFT data (different from standard SFT, as pretraining on SFT data means no masking on input prompts) while evaluating on SFT data with labels masked for input prompts, following the mixed training approach from https://arxiv.org/pdf/2309.14316.pdf
                if dataset_config.mixed_training and dataset_config.pretrain is False:
                    logging.warning(f"SFT Training Set {dataset_train.dataset_config.dataset}: setting labels the same as input_ids for mixed training, because {dataset_config.mixed_training=}...")
                    dataset_train.hf_dataset = dataset_train.hf_dataset.map(lambda x: {"labels": x["input_ids"]}, num_proc=dataset_config.max_workers)

                self.dataset_train.append(dataset_train)
            except Exception as e:
                logging.exception(f"Failed to load and preprocess the dataset {dataset_config.dataset}. Error: {e}")
                logging.info(f"Skipping the dataset {dataset_config.dataset} and continue to the next dataset...")
                continue

        # Merge all the datasets into a single dataset
        logging.info(f"Merging all the datasets {[dataset.dataset_config.dataset for dataset in self.dataset_train]} into a single dataset...")
        self.dataset_train = MergedDataset(self.configs, self.dataset_train, split="train")
        logging.info(f"--> Total Training Set Length = {len(self.dataset_train)}. Total Token count = {self.dataset_train.num_tokens}. Max token length = {max(self.dataset_train.lengths) if hasattr(self.dataset_train, 'lengths') and self.dataset_train.lengths else 'N/A'}. Min token length = {min(self.dataset_train.lengths) if hasattr(self.dataset_train, 'lengths') and self.dataset_train.lengths else 'N/A'}")
        if self.dataset_val:
            self.dataset_val = MergedDataset(self.configs, self.dataset_val, split="test")
            logging.info(f"--> Total Validation Set Length = {len(self.dataset_val)}. Total Token count = {self.dataset_val.num_tokens}. Max token length = {max(self.dataset_val.lengths) if hasattr(self.dataset_val, 'lengths') and self.dataset_val.lengths else 'N/A'}. Min token length = {min(self.dataset_val.lengths) if hasattr(self.dataset_val, 'lengths') and self.dataset_val.lengths else 'N/A'}")

        train_dl_kwargs = self.get_dataloader_kwargs(self.dataset_train, "train")

        # Create DataLoaders for the training and validation dataset
        self.train_dataloader = DataLoader(
            self.dataset_train,
            num_workers=self.configs.train_config.num_workers_dataloader,
            pin_memory=True,
            **train_dl_kwargs,
        )

        self.eval_dataloader = None
        if self.configs.train_config.run_validation:
            val_dl_kwargs = self.get_dataloader_kwargs(self.dataset_val, "val")

            self.eval_dataloader = DataLoader(
                self.dataset_val,
                num_workers=self.configs.train_config.num_workers_dataloader,  # set it to 0 due to https://github.com/pytorch/pytorch/issues/8976, may not worth risking with more than 0 for validation
                pin_memory=True,
                **val_dl_kwargs,
            )

    def get_preprocessed_dataset(self, dataset_config: AbstractDataset, split: str = "train") -> torch.utils.data.Dataset:
        logging.info(f"Loading and preprocessing the dataset: {dataset_config.dataset} - {split}...")
        # Check if the dataset has been preprocessed and cached, save the effort if yes, only when force_refresh is False
        processed_dataset = None

        # take the model name as the cache folder, if it is a path, take the last part, if it is a model name, as is
        model_name_or_path = self.configs.tokenizer_loading_config.pretrained_model_name_or_path
        model_name = Path(model_name_or_path).name if Path(model_name_or_path).exists() else model_name_or_path
        # add model name in the cache path
        dataset_cache_dir = Path(dataset_config.dataset_cache_dir) / model_name 
        dataset_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
        # add split in the cache path
        dataset_cache_dir = dataset_cache_dir / split
        dataset_cache_dir.resolve().mkdir(parents=True, exist_ok=True)        
        # add batch strategy in the cache path
        dataset_cache_dir = dataset_cache_dir / self.configs.train_config.batching_strategy
        dataset_cache_dir.resolve().mkdir(parents=True, exist_ok=True)

        # try to load the dataset from cache if force_refresh is False
        if dataset_config.force_refresh is False:
            processed_dataset = self.get_dataset_from_cache(dataset_config, dataset_cache_dir)
        else:
            processed_dataset = None
            logging.info(f"Force refresh is set to True, reprocessing the dataset. Cache path: {dataset_cache_dir}..")
        if processed_dataset is not None:
            return processed_dataset

        # Sync through the barrier to avoid duplicated work
        if self.configs.setup_config.rank == 0:
            processed_dataset = self.process_dataset(dataset_config, split)
            # Save the processed dataset to the cache path, FIXME: metadata is not persisted
            processed_dataset.save_to_disk(dataset_cache_dir)  # type: ignore
            logging.info("Finished text preprocessing from the main process")
            # Signal other Processes to wait until the main process has completed the mapping and saving
            torch.distributed.barrier()
        else:
            logging.info("Waiting for main process to perform text preprocessing")
            torch.distributed.barrier()
            logging.info("Finished waiting for main process to perform text preprocessing, loading from cache...")
            processed_dataset = self.get_dataset_from_cache(dataset_config, dataset_cache_dir)
            logging.info("Finished loading from cache")
            if processed_dataset is None:
                logging.warning("[abnormal] The dataset is not yet cached by main process in {dataset_cache_dir}. Reprocessing in other rank...")         
                processed_dataset = self.process_dataset(dataset_config, split)

        return processed_dataset

    def process_dataset(self, dataset_config: AbstractDataset, split: str = "train"):
        def get_split_path() -> str:
            return (
                dataset_config.train_split
                if split == "train"
                else dataset_config.test_split
            )

        if dataset_config.dataset not in DATASET_PREPROC:
            raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

        logging.info(f"Processing the dataset {dataset_config.dataset} - {split}...")
        # Set the environment variable to disable the network access for huggingface datasets
        os.environ['HF_DATASETS_OFFLINE'] = "1"
        processed_dataset = DATASET_PREPROC[dataset_config.dataset](
            dataset_config,
            self.tokenizer,
            get_split_path()
        )
        logging.info("Finished processing the dataset.")

        logging.info(f"Applying batching strategy {self.configs.train_config.batching_strategy}...")
        logging.info(f"Dataset {dataset_config.dataset} - {split} length before batching: {len(processed_dataset)}")
        # If the batching strategy is packing, concatenate the tokenized samples into long sequences filling up the context length of the model
        processed_dataset = BatchDataset(train_config=self.configs.train_config,
                                         dataset_config=dataset_config,
                                         hf_dataset=processed_dataset,
                                         tokenizer=self.tokenizer,
                                         load_from_disk=False,
                                         split=split)
        logging.info("Finished applying batching strategy.")

        return processed_dataset

    def get_dataset_from_cache(self, dataset_config: AbstractDataset, dataset_cache_dir: str) -> Optional[torch.utils.data.Dataset]:
        # Load the dataset from cache
        if os.path.exists(dataset_cache_dir):
            # Load the dataset from the specified path, FIXME: metadata is not persisted so not loaded
            logging.warning(f"Loading dataset from cache folder: {dataset_cache_dir}...")
            try:
                processed_dataset = BatchDataset.load_from_disk(
                    train_config=self.configs.train_config,
                    dataset_config=dataset_config,
                    dataset_cache_dir=dataset_cache_dir
                )
            except Exception as e:
                logging.warning(f"Failed to load dataset from cache folder: {dataset_cache_dir}. Error: {e}")
                return None
            logging.warning("Dataset successfully loaded from disk.")
            return processed_dataset
        else:
            logging.info(f"The dataset is not yet cached in {dataset_cache_dir}.")
            return None

    def get_dataloader_kwargs(self, dataset: torch.utils.data.Dataset, mode: str = "train") -> Dict[str, Any]:
            kwargs = {}
            batch_size = self.configs.train_config.per_device_train_batch_size if mode=="train" else self.configs.train_config.per_device_eval_batch_size
            if self.configs.train_config.batching_strategy == "padding":
                if self.configs.train_config.enable_fsdp or self.configs.train_config.enable_deepspeed:
                    kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                        dataset=dataset,
                        train_config=self.configs.train_config,
                        batch_size=batch_size,
                        num_replicas=self.configs.setup_config.world_size,
                        rank=self.configs.setup_config.rank,
                        shuffle=mode=="train",
                        seed=self.configs.setup_config.random_seed,
                    )
                else:
                    # LengthBasedBatchSampler(dataset, batch_size, drop_last=self.configs.train.config.drop_last, shuffle=mode=="train")      
                    kwargs["batch_sampler"] = LengthBasedBatchSampler(
                        dataset=dataset,
                        train_config=self.configs.train_config,
                        batch_size=batch_size,
                        shuffle=mode=="train",
                        seed=self.configs.setup_config.random_seed,
                    )
                # DataCollatorForSeq2Seq will dynamically pad the inputs received with similar length, as well as the labels.
                kwargs["collate_fn"] = DataCollatorForSeq2Seq(self.tokenizer)
            elif self.configs.train_config.batching_strategy == "packing":
                if self.configs.train_config.enable_fsdp or self.configs.train_config.enable_deepspeed:
                    kwargs["sampler"] = CustomDistributedSampler(
                    dataset=dataset,
                    batch_size=batch_size,
                    rank=self.configs.setup_config.rank,
                    num_replicas=self.configs.setup_config.world_size,
                    shuffle=mode=="train",
                )
                kwargs["batch_size"] = batch_size
                kwargs["drop_last"] = self.configs.train_config.drop_last
                kwargs["collate_fn"] = default_data_collator
            else:
                raise ValueError(f"Unknown batching strategy: {self.configs.train_config.batching_strategy}")

            return kwargs


class BatchDataset(Dataset):
    # this is a wrapper class for the hf_dataset, which is batched according to the batching_strategy
    # list of important attributes: hf_dataset (datasets.Dataset), context_length, num_proc, batching_strategy
    def __init__(self, train_config: TrainConfigCommon, dataset_config: AbstractDataset, hf_dataset: Optional[HFDataset]=None, phase_batches: Optional[HFDataset]=None, tokenizer=None, load_from_disk: bool=False, split: str="train"):
        self.hf_dataset: HFDataset = hf_dataset  # type: ignore
        self.phase_batches: HFDataset = phase_batches  # type: ignore
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.dataset_config = dataset_config
        self.split = split

        # batch hf_dataset by batching_strategy. If load_from_disk is True, the dataset is already batched and saved to disk
        if not load_from_disk:
            self.batch_hf_dataset()

    def batch_hf_dataset(self):
        """Batch the hf_dataset according to the batching_strategy."""
        def process_batch(batch):
            # If the batching strategy is packing, concatenate the tokenized samples into long sequences filling up the context length of the model, else computing lengths for batching with similar length with padding.
            if self.train_config.batching_strategy == "padding":
                # Note that we don't actually pad here
                # DataCollatorForSeq2Seq of dataloading's Data collator will dynamically pad the inputs received with similar length, as well as the labels.
                # pre-compute the length of each example for merge sort batching, note that batch["lengths"] is required for fixed batch size, so it is always computed separately before the phase_batches
                first_key = next(iter(batch.keys()))
                lengths = [len(d) for d in batch[first_key]]
                batch["lengths"] = lengths

                return batch
            else:  # packing
                # Initialize packed_samples with empty lists for each key, list be filled with self.train_config.context_length number of int
                packed_samples = {key: [] for key in batch.keys()}

                for key in batch.keys():
                    # Convert list of lists to a tensor for concatenation
                    concatenated = torch.cat([torch.tensor(sublist) for sublist in batch[key]], dim=0)
                    current_length = concatenated.size(0)

                    start_idx = 0
                    while current_length > self.train_config.context_length:
                        end_idx = start_idx + self.train_config.context_length
                        # Slice tensor and append it to packed_samples, converting tensor slice to list
                        packed_samples[key].append(concatenated[start_idx:end_idx].tolist())
                        start_idx = end_idx
                        current_length -= self.train_config.context_length

                    # Handle leftovers in the buffer for each key, either drop it or pad it
                    if not self.dataset_config.drop_last and current_length > 0:
                        # Get the remaining elements of concatenated
                        remaining_elements = concatenated[start_idx:]

                        # Determine the padding length
                        padding_length = self.train_config.context_length - len(remaining_elements)

                        # Create the padding tensor
                        if key == "input_ids":
                            padding_value = self.tokenizer.pad_token_id
                        elif key == "labels":
                            padding_value = -100
                        elif key == "attention_mask":
                            padding_value = 0
                        else:
                            padding_value = 0  # or whatever value you want to use for padding

                        padding = [padding_value] * padding_length

                        # Pad the remaining elements and append to packed_samples(left padding)
                        packed_samples[key].append((padding + remaining_elements.tolist()))

                return packed_samples

        # Calculate batch_size for parallel processing within .map
        # Note: better self.dataset_config.num_batches 2~4 times of self.dataset_config.max_workers to save cpu memory for large dataset
        batch_size_for_map = max(1, len(self.hf_dataset) // self.dataset_config.num_batches)

        # Apply batching function using .map, hf_dataset is batched with ["input_ids"], ["attention_mask"], ["labels"] keys.
        # For padding, hf_dataset has additional key "lengths" for merge sort batching
        self.hf_dataset = self.hf_dataset.map(
            function=process_batch,
            batched=True,
            batch_size=batch_size_for_map,
            load_from_cache_file=False if self.dataset_config.force_refresh else True,
            num_proc=self.dataset_config.max_workers,
            desc=f"Applying batching with {self.train_config.batching_strategy} strategy: {'computing sequence lengths' if self.train_config.batching_strategy == 'padding' else 'packing sequences to fixed length'}"
        )

        # Apply phase batching for dynamic batch size, w/wo curriculum learning if self.train_config.dynamic_batch_size is True and self.train_config.curriculum_learning is True
        # Apply fixed batch size with merge sort batching otherwise
        if self.train_config.precompute_batches and self.train_config.precompute_batches == "per_dataset":
            logging.info(f"{self.train_config.dynamic_batch_size=} and {self.train_config.curriculum_learning=}, precompute batches {'with' if self.train_config.curriculum_learning else 'without'} phase batching...")

            # Add a index column to the hf_dataset for phase batching, need to keep track of the original index for each data point
            # because multi-processing of creating dynamic batch of original index based on each data points length, each batch should be a list of original indices of the data points
            logging.info("Adding 'indices' column to the hf_dataset for batching...")
            self.hf_dataset = self.hf_dataset.add_column("indices", list(range(len(self.hf_dataset))))

            self.phase_batches = self.hf_dataset.map(
                function=partial(precompute_phased_batch, train_config=self.train_config, split=self.split),
                batched=True,
                batch_size=batch_size_for_map,
                load_from_cache_file=False if self.dataset_config.force_refresh else True,
                num_proc=self.dataset_config.max_workers,
                remove_columns=list(self.hf_dataset.features),
                desc=f"[{self.split}] Computing batches - dynamic_batch_size: {self.train_config.dynamic_batch_size} curriculum_learning: {self.train_config.curriculum_learning}"
            )

            # remove column 'indices' from batch_dataset.hf_dataset insteads of adjusting it as it is only required for debugging purpose at this point
            logging.info("Removing 'indices' column from the hf_dataset as batching is done...")
            self.hf_dataset = self.hf_dataset.remove_columns("indices")

    def save_to_disk(self, dataset_cache_dir: str):
        """Save the hf_dataset to disk."""
        logging.info(f"Saving final transformed dataset to cache folder: {dataset_cache_dir}...")
        self.hf_dataset.save_to_disk(dataset_cache_dir)

        if self.phase_batches:
            phase_batches_cache_dir = Path(dataset_cache_dir) / "phase_batches"
            phase_batches_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
            self.phase_batches.save_to_disk(phase_batches_cache_dir)

    @classmethod
    def load_from_disk(cls, train_config: TrainConfigCommon, dataset_config: AbstractDataset, dataset_cache_dir: str):
        """Load the hf_dataset from disk and return a new BatchDataset instance."""
        logging.info(f"Loading dataset from cache folder: {dataset_cache_dir}...")
        hf_dataset: HFDataset = HFDataset.load_from_disk(dataset_cache_dir)

        phase_batches_cache_dir = Path(dataset_cache_dir) / "phase_batches"
        phase_batches: HFDataset = None
        if phase_batches_cache_dir.exists():
            phase_batches = HFDataset.load_from_disk(phase_batches_cache_dir) if os.path.exists(phase_batches_cache_dir) else None

        return cls(train_config, dataset_config, hf_dataset=hf_dataset, phase_batches=phase_batches, load_from_disk=True)

    def train_test_split(self, test_size: float, seed: int = 42, **kwargs):
        """Split the dataset into training and testing sets."""
        splited_dataset: DatasetDict = self.hf_dataset.train_test_split(test_size=test_size, seed=seed, **kwargs)
        logging.info(f"Splitting dataset {self.dataset_config.dataset} - {self.split} into training and testing sets with test_size={test_size} and seed={seed}...")
        # TODO: handle self.phase_batches, current implementation does not work with precompute_batches per_dataset if the dataset needs to be split by test_split_ratio
        return BatchDataset(self.train_config, self.dataset_config, hf_dataset=splited_dataset["train"], load_from_disk=True, split="Train"), BatchDataset(self.train_config, self.dataset_config, hf_dataset=splited_dataset["test"], load_from_disk=True, split="Test")

    @property
    def num_tokens(self):
        if self.train_config.batching_strategy == "padding":
            return sum(self.hf_dataset["lengths"])
        else:
            return len(self.hf_dataset) * self.train_config.context_length

    @property
    def lengths(self):
        if "lengths" in self.hf_dataset.column_names:
            return self.hf_dataset["lengths"]
        else:
            return None

    def __getitem__(self, idx):
        return self.hf_dataset[int(idx)]

    def __getitems__(self, idxs):
        batch = self.hf_dataset[idxs]
        if isinstance(batch, dict):
            return [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        else:
            return self.hf_dataset[idxs]

    def __len__(self):
        return len(self.hf_dataset)


class MergedDataset(Dataset):
    # this is a wrapper class for the concatenated_datasets used for training and evaluation for better debuggable code within the trainer
    def __init__(self, configs: Configs, datasets: List[BatchDataset], split: str = "train"):
        self.configs = configs  # for debugging
        self.split = split
        self.hf_dataset, phase_batches = self._concatenate_datasets(datasets)

        if "lengths" in self.hf_dataset.column_names:
            self.lengths = self.hf_dataset['lengths']
            # remove lengths from the dataset
            self.hf_dataset = self.hf_dataset.remove_columns("lengths")
        else:
            self.lengths = None

        # check if there are phase numbers in hf_dataset.column_names, it can be 1, 2, 3, depending on the number of phases in curriculum learning defined by self.configs.train_config.curriculum_phases
        if phase_batches:
            self.phase_batches: Dict[int, List[List[int]]] = {}
            # phase_batches["batches"]  # list of list of indices of the data points
            # phase_batches["phase"]  # list of phase number for each batch, put a phase tag
            # convert it back to dict of key: phase number, value: list of batches with dynamic batch size, each batch is a list of indices of the data points
            for phase, batch in zip(phase_batches["phase"], phase_batches["batches"]):
                self.phase_batches.setdefault(phase, []).append(batch)

    def __getitem__(self, idx):
        return self.hf_dataset[int(idx)]

    def __getitems__(self, idxs):
        batch = self.hf_dataset[idxs]
        if isinstance(batch, dict):
            return [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        else:
            return batch

    def __len__(self):
        return len(self.hf_dataset)

    @property
    def num_tokens(self):
        if self.lengths:
            return sum(self.lengths)
        else:
            return len(self.hf_dataset) * self.configs.train_config.context_length  # assuming it is packed by train_config.batching_strategy == "packing"

    def _concatenate_datasets(self, batch_datasets: List[BatchDataset]) -> (HFDataset, HFDataset):
        if not batch_datasets:  # at least one dataset
            return ValueError(f"No datasets to concatenate {batch_datasets}")

        hf_dataset = concatenate_datasets([dataset.hf_dataset for dataset in batch_datasets])

        if self.configs.train_config.precompute_batches:
            if self.configs.train_config.precompute_batches == "per_dataset":
                # prepare the phase_batches of each dataset before merging
                logging.info("Adjusting indices for precomputed batches phase_batches for each dataset to be merged ...")
                adjusted = True
                current_indices = len(batch_datasets[0])
                for batch_dataset in batch_datasets[1:]:
                    if(not self._adjust_indices(batch_dataset, current_indices)):
                        adjusted = False
                    current_indices += len(batch_dataset)

                if adjusted:
                    phase_batches = concatenate_datasets([dataset.phase_batches for dataset in batch_datasets])
                else:
                    phase_batches = None
                    logging.warning(f"{self.configs.train_config.precompute_batches=}. But indices are not adjusted for phase_batches or hf_dataset[]indices'], likely because it is not cached properly."
                                    "Setting phase_batches to None to recompute phase_batches for the merged dataset.")
            elif self.configs.train_config.precompute_batches == "combined":
                logging.info(f"{self.configs.train_config.dynamic_batch_size=} and {self.configs.train_config.curriculum_learning=}, precompute batches {'with' if self.configs.train_config.curriculum_learning else 'without'} phase batching...")

                # Add a index column to the hf_dataset for phase batching, need to keep track of the original index for each data point
                # because multi-processing of creating dynamic batch of original index based on each data points length, each batch should be a list of original indices of the data points
                logging.info("Adding 'indices' column to the hf_dataset for batching...")
                hf_dataset = hf_dataset.add_column("indices", list(range(len(hf_dataset))))

                def precompute_phased_batch_combined(force_refresh: bool = False, max_workers: int = 1):
                    batch_size_for_map = max(1, len(hf_dataset) // self.configs.dataset_configs[0].num_batches)
                    phase_batches = hf_dataset.map(
                        function=partial(precompute_phased_batch, train_config=self.configs.train_config, split=self.split),
                        batched=True,
                        batch_size=batch_size_for_map,
                        load_from_cache_file=False if force_refresh else True,
                        num_proc=max_workers,
                        remove_columns=list(hf_dataset.features),
                        desc=f"[{self.split}] Computing batches - dynamic_batch_size: {self.configs.train_config.dynamic_batch_size} curriculum_learning: {self.configs.train_config.curriculum_learning}"
                    )
                    return phase_batches

                phase_batches = precompute_phased_batch_combined(force_refresh=True)

                # remove column 'indices' from batch_dataset.hf_dataset insteads of adjusting it as it is only required for debugging purpose at this point
                logging.info("Removing 'indices' column from the hf_dataset as batching is done...")
                hf_dataset = hf_dataset.remove_columns("indices")
            else:
                raise ValueError(f"Unknown precompute_batches: {self.configs.train_config.precompute_batches}")
        else:
            phase_batches = None

        return hf_dataset, phase_batches

    def _adjust_indices(self, batch_dataset: BatchDataset, base_idx: int=0, max_workers: int=1) -> bool:
        """
        Adjust the indices of the hf_dataset and phase_batches to the new indices by adding base_idx. This is to prepare the merge of datasets with global unique indices(avoid overlapping indices).
        e.g. hf_dataset 1: 0~999, hf_dataset 2: 0~888, ... after merge, hf_dataset 1: 0~999, hf_dataset 2: 1000~1888, ... so batch_dataset.phase_batches needs to be adjusted accordingly

        phase_batches has column "batches" which is the list of list of original indices of the data points, need to adjust it to the new indices too.
        e.g. phase_batches["batches"] = [[0, 1, 2], [3, 4, 5], ...] -> [[1000, 1001, 1002], [1003, 1004, 1005], ...] if base_idx = 1000

        return True if phase_batches is adjusted, False otherwise
        """
        if "batches" in batch_dataset.phase_batches.column_names and batch_dataset.phase_batches and batch_dataset.train_config.precompute_batches == "per_dataset":
            # adjust the indices in phase_batches
            batch_dataset.phase_batches = batch_dataset.phase_batches.map(
                lambda examples: {'batches': [idx + base_idx for idx in examples['batches']]},
                num_proc=max_workers,
                cache_file_name=str(Path(batch_dataset.dataset_config.dataset_map_cache_dir) / f"merge_phase_batches_reindexing_{base_idx}.arrow"),
                load_from_cache_file=False if batch_dataset.dataset_config.force_refresh else True,
                desc=f"[{self.split}] Adjusting phase_batches indices - {batch_dataset.dataset_config.dataset}"
            )
            return True
        else:
            return False

def precompute_phased_batch(batch, train_config: TrainConfigCommon, split: str="train") -> Dict[str, Any]:
    if "lengths" not in batch.keys():  # job may fail in the middle, so check if "lengths" is already computed
        # pre-compute the length of each example for merge sort batching
        first_key = next(iter(batch.keys()))
        lengths = [len(d) for d in batch[first_key]]
    else:
        lengths = batch["lengths"]

    indices = batch["indices"]

    # pre-compute the phase_batches which is dict of key: phase number, value: list of batches with dynamic batch size, each batch is a list of indices of the data points
    # note that this is Single-Dataset Mini-Batches: mini-batches from a single dataset at a time. TODO: Mixed Mini-Batches: each mini-batch contains data from multiple datasets
    batch_size = train_config.per_device_train_batch_size if split=="train" else train_config.per_device_eval_batch_size
    phase_batches: Dict[int, List[List[int]]] = create_batches(train_config=train_config, lengths=lengths, indices=indices, batch_size=batch_size, debug=train_config.debug)
    # convert phase_batches to flat List[List[int]] by merging batches in all phases
    batches = {}
    batches["batches"] = [batch for phase in phase_batches for batch in phase_batches[phase]]  # list of list of indices of the data points
    batches["phase"] = [phase for phase in phase_batches for _ in phase_batches[phase]]  # list of phase number for each batch, put a phase tag

    return batches
