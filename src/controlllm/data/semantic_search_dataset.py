import random
import logging
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from typing import List, Dict

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from datasets import Dataset as HFDataset, Features, Value, Sequence, load_from_disk, concatenate_datasets
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from controlllm.configs.datasets import AbstractDataset as DatasetConfig
from controlllm.data.utils import load_files_as_dataset, sample_dataset


class SemanticSearch(Dataset):
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
            self.hf_dataset: HFDataset = load_files_as_dataset(
                data_dir=self.data_path, cache_dir=self.dataset_raw_cache_dir, max_workers=self.dataset_config.max_workers)
            # check if the dataset is loaded
            if self.hf_dataset is None:
                raise ValueError(f"No data loaded from path: {self.data_path}")
            logging.info(f"Finished loading data from path: {self.data_path}")
            if self.dataset_config.sample_before_preprocessing:
                logging.info(f"Sampling the dataset: {self.dataset_config.dataset} - {self.data_path}")
                self.hf_dataset = sample_dataset(self.hf_dataset, self.dataset_config, data_path)
                logging.info(f"Finished sampling the dataset: {self.dataset_config.dataset} - {self.data_path}")
            else:
                logging.info(f"Sample before preprocessing is turned off for the dataset: {self.dataset_config.dataset} - {self.data_path}")
            logging.info(f"Converting the dataset to features: {self.dataset_config.dataset}")
            self.convert_to_features()
            logging.info(f"Finished converting the dataset to features: {self.dataset_config.dataset}")
        except Exception as e:
            logging.exception(f"Loading of dataset failed!: {e}")
            raise

    def __len__(self):
        return self.hf_dataset.shape[0]

    # TODO: Add text cleaning by hf_dataset transform(e.g. mapping)
    def _apply_prompt_template(self, batch, K=0):
        """
        Map function that processes one batch of size 1 at a time.

        - If K=0, set permute=False, returns a SINGLE row (dictionary of length-1 lists).
        - If K>0 or K=-1, set permute=True, returns MULTIPLE rows (dictionary of length-K lists),
        one for each permutation or random sample of permutations.

        This is done so that Hugging Face expands those lists into multiple rows.
        """
        permute = K != 0  # if K=0, don't permute

        # Each field in `batch` is a list of length == batch_size (which we'll set to 1).
        # So we extract the single sample's value. For example:
        sample = {}
        for key, values in batch.items():
            sample[key] = values[0]  # take the first (and only) item

        # Data setup
        group_id = str(sample[self.dataset_config.group_column])
        relevance_score = float(sample[self.dataset_config.relevance_column])

        doc_cols = self.dataset_config.response_column
        query_cols = self.dataset_config.prompt_columns

        # We'll accumulate the final prompts/responses in these lists
        prompts = []
        responses = []
        relevance_scores = []
        group_ids = []

        # -----------------------------------------
        # CASE 1: No permutations => exactly 1 output row
        # -----------------------------------------
        if not permute:
            input_text = "".join(
                f"# {col}: {sample[col] if col in sample else 'N/A'}\n"
                for col in doc_cols
            )
            response_ = f'{input_text}'
            # prompt_ = f'Query: {sample[query_cols]}'  # query is tagged with prefix
            query_col = query_cols[0]  # else query_cols means if it is configured as a string

            prompt_ = f'Query: {sample[query_col]}'  # query is tagged with prefix

            # Append to our lists (one item)
            prompts.append(prompt_)
            responses.append(response_)
            relevance_scores.append(relevance_score)
            group_ids.append(group_id)

            # Return a dict of lists, each list has length=1
            return {"group_id": group_ids, "prompt": prompts, "response": responses, "relevance_score": relevance_scores}

        # -----------------------------------------
        # CASE 2: Permutations => multiple output rows
        # -----------------------------------------
        # If K != -1, pick K random permutations. Otherwise, use all permutations.
        all_perms = list(itertools.permutations(doc_cols))
        if K != -1 and K < len(all_perms):
            chosen_perms = random.sample(all_perms, K)
        else:
            chosen_perms = all_perms

        # Build one row for each permutation
        for permutation in chosen_perms:
            input_text = "".join(
                f"# {col}: {sample[col] if col in sample else 'N/A'}\n"
                for col in permutation
            )
            response_ = f'{input_text}'
            # prompt_ = f'Query: {sample[query_cols]}'  # query is tagged with prefix
            query_col = query_cols[0]  # else query_cols means if it is configured as a string

            prompt_ = f'Query: {sample[query_col]}'  # query is tagged with prefix

            prompts.append(prompt_)
            responses.append(response_)
            relevance_scores.append(relevance_score)
            group_ids.append(group_id)

        # Return a dict of lists (length=K). HF will create K rows automatically.
        return {"group_id": group_ids, "prompt": prompts, "response": responses, "relevance_score": relevance_scores}

    def _group_by_group_id(self, dataset: HFDataset, group_column: str) -> HFDataset:
        """
        Groups the dataset by group_column.
        Produces one row per group with values for "prompt", "response" and "relevance_score" as lists.
        """
        # Convert dataset to a dictionary
        data = dataset.to_dict()
        groups = {}
        # Iterate over each row in the dataset to accumulate group data
        for i, gid in enumerate(tqdm(data[group_column], desc="Processing groups")):
            if gid not in groups:
                groups[gid] = {
                    group_column: gid,
                    "prompt": [],
                    "response": [],
                    "relevance_score": []
                }
            groups[gid]["prompt"].append(data["prompt"][i])
            groups[gid]["response"].append(data["response"][i])
            groups[gid]["relevance_score"].append(data["relevance_score"][i])

        # Prepare a dictionary of lists for HF Datasets
        grouped_data = {
            group_column: [],
            "prompts": [],
            "responses": [],
            "relevance_scores": []
        }
        for gid, row in groups.items():
            grouped_data[group_column].append(gid)
            grouped_data["prompts"].append(row["prompt"])
            grouped_data["responses"].append(row["response"])
            grouped_data["relevance_scores"].append(row["relevance_score"])

        features = Features({
            group_column: Value("string"),
            "prompts": Sequence(Value("string")),
            "responses": Sequence(Value("string")),
            "relevance_scores": Sequence(Value("string")),
        })

        return HFDataset.from_dict(grouped_data, features)

    def _sample_pairs_from_groups(self, batch, K=5) -> dict:
        """
        For each group row in the batch (each with list values for "prompt", "response", "relevance_score"):
        - Convert the group's scores to float.
        - Sort the group's rows by relevance_score in ascending order.
        - Generate all unique pairs (i, j) with i < j using broadcasting (excluding self-pairs and pairs with equal scores).
        * For each pair, use the lower scored row's prompt and response as the "reject" 
            and the higher scored row's response as the "chosen" response.
        * Compute margin = score[j] - score[i].
        - If the number of pairs exceeds K for that group, randomly sample K pairs.

        Returns a dict with lists for:
        - "prompt": from the lower scored row,
        - "response_chosen": chosen responses (higher score),
        - "response_reject": reject responses (lower score),
        - "margin": score difference,
        - "group_id": the associated group id,
        - "score_chosen": chosen score,
        - "score_reject": reject score.
        """
        prompts_out = []
        responses_chosen_out = []
        responses_reject_out = []
        margins_out = []
        group_ids_out = []
        scores_chosen_out = []
        scores_reject_out = []

        n_rows = len(batch["group_id"])
        for idx in range(n_rows):
            gid = batch["group_id"][idx]
            group_prompts = batch["prompts"][idx]         # list of prompts
            group_responses = batch["responses"][idx]       # list of responses
            # Convert all scores in the group to float immediately
            group_scores = [float(x) for x in batch["relevance_scores"][idx]]   # list of float scores

            # Skip empty groups
            if not group_prompts:
                continue

            # Sort indices by relevance score (ascending)
            sorted_indices = sorted(range(len(group_scores)), key=lambda i: group_scores[i])
            sorted_prompts = [group_prompts[i] for i in sorted_indices]
            sorted_responses = [group_responses[i] for i in sorted_indices]
            sorted_scores = [group_scores[i] for i in sorted_indices]

            # Convert scores to a NumPy array for broadcasting
            scores_array = np.array(sorted_scores, dtype=np.float32)  # shape: (m,)
            m = len(scores_array)

            # Compute pairwise differences: margin_matrix[i, j] = score[j] - score[i]
            margin_matrix = scores_array[None, :] - scores_array[:, None]

            # Get indices for all pairs with i < j (excludes diagonal self-pairs)
            i_indices, j_indices = np.triu_indices(m, k=1)

            # Prepare a temporary list of pairs for the current group
            group_pairs = []
            for i, j in zip(i_indices, j_indices):
                # Only add pair if the scores are different (margin > 0)
                if margin_matrix[i, j] > 0:
                    prompt_val = sorted_prompts[i]         # prompt from lower scored row
                    chosen_response = sorted_responses[j]    # higher scored row's response
                    reject_response = sorted_responses[i]    # lower scored row's response
                    margin = margin_matrix[i, j]             # score difference (should be > 0)
                    score_chosen = sorted_scores[j]
                    score_reject = sorted_scores[i]
                    group_pairs.append((prompt_val, chosen_response, reject_response, margin, gid, score_chosen, score_reject))

            # If there are more pairs than K, randomly sample K of them for this group
            if K != -1 and len(group_pairs) > K:
                group_pairs = random.sample(group_pairs, K)

            # Append the pairs from this group to the overall output lists
            for tup in group_pairs:
                prompts_out.append(tup[0])
                responses_chosen_out.append(tup[1])
                responses_reject_out.append(tup[2])
                margins_out.append(tup[3])
                group_ids_out.append(tup[4])
                scores_chosen_out.append(tup[5])
                scores_reject_out.append(tup[6])
    
        return {
            "prompt": prompts_out,
            "response_chosen": responses_chosen_out,
            "response_reject": responses_reject_out,
            "margin": margins_out,
            "group_id": group_ids_out,
            "score_chosen": scores_chosen_out,
            "score_reject": scores_reject_out
        }

    def _tokenize_add_label(self, sample, add_special_tokens=True):
        # don't do padding here, it will be done in the dataloader. don't need to do truncation if packing is true
        prompt_input_ids = self.tokenizer.encode(sample["prompt"],
                                            truncation=self.dataset_config.truncation,
                                            max_length=self.dataset_config.max_length,
                                            add_special_tokens=False)

        chosen_input_ids = self.tokenizer.encode(sample["response_chosen"],
                                            truncation=self.dataset_config.truncation,
                                            max_length=self.dataset_config.max_length,
                                            add_special_tokens=False)

        rejected_input_ids = self.tokenizer.encode(sample["response_reject"],
                                            truncation=self.dataset_config.truncation,
                                            max_length=self.dataset_config.max_length,
                                            add_special_tokens=False)

        # Add special tokens if needed
        prompt_attention_mask = [1] * len(prompt_input_ids)
        chosen_attention_mask = [1] * len(chosen_input_ids)
        rejected_attention_mask = [1] * len(rejected_input_ids)
        if add_special_tokens:
            if self.tokenizer.bos_token_id is not None:
                prompt_input_ids = [self.tokenizer.bos_token_id] + prompt_input_ids
                # According to the llama paper, better to mask the special token for contrastive learning. TODO: check by experiment
                prompt_attention_mask = [1] + prompt_attention_mask
                chosen_input_ids = [self.tokenizer.bos_token_id] + chosen_input_ids
                chosen_attention_mask = [1] + chosen_attention_mask
                rejected_input_ids = [self.tokenizer.bos_token_id] + rejected_input_ids
                rejected_attention_mask = [1] + rejected_attention_mask
            if self.tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [self.tokenizer.eos_token_id]
                prompt_attention_mask = prompt_attention_mask + [1]
                chosen_input_ids = chosen_input_ids + [self.tokenizer.eos_token_id]
                chosen_attention_mask = chosen_attention_mask + [1]
                rejected_input_ids = rejected_input_ids + [self.tokenizer.eos_token_id]
                rejected_attention_mask = rejected_attention_mask + [1]

        sample = {
            "prompt_input_ids": prompt_input_ids,  # follow sentence-transformers's convention
            "chosen_input_ids": chosen_input_ids,  # follow sentence-transformers's convention
            "rejected_input_ids": rejected_input_ids,  # follow sentence-transformers's convention
            "prompt_attention_mask": prompt_attention_mask,  # follow sentence-transformers's convention
            "chosen_attention_mask": chosen_attention_mask,  # follow sentence-transformers's convention
            "rejected_attention_mask": rejected_attention_mask,  # follow sentence-transformers's convention
            "label": sample["margin"],  # follow sentence-transformers's convention
            "group_id": sample["group_id"],
            "score_chosen": sample["score_chosen"],
            "score_reject": sample["score_reject"]
        }
        return sample

    @classmethod
    def _precompute_response_embeddings(cls, batch, tokenizer, model):
        """
        Precompute embeddings for the response pairs in 'batch'.
        With sharded data, each rank processes its own batch.
        """
        # Unpack and pad sequences
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        rejected_attention_mask = batch["rejected_attention_mask"]

        max_len_chosen = max(len(ids) for ids in chosen_input_ids)
        chosen_input_ids = [
            [tokenizer.pad_token_id] * (max_len_chosen - len(ids)) + ids
            for ids in chosen_input_ids
        ]
        chosen_attention_mask = [
            [0] * (max_len_chosen - len(mask)) + mask
            for mask in chosen_attention_mask
        ]

        max_len_rejected = max(len(ids) for ids in rejected_input_ids)
        rejected_input_ids = [
            [tokenizer.pad_token_id] * (max_len_rejected - len(ids)) + ids
            for ids in rejected_input_ids
        ]
        rejected_attention_mask = [
            [0] * (max_len_rejected - len(mask)) + mask
            for mask in rejected_attention_mask
        ]

        # Convert to tensors on model.device
        chosen_input_ids = torch.tensor(chosen_input_ids, dtype=torch.long, device=model.device)
        rejected_input_ids = torch.tensor(rejected_input_ids, dtype=torch.long, device=model.device)
        chosen_attention_mask = torch.tensor(chosen_attention_mask, dtype=torch.long, device=model.device)
        rejected_attention_mask = torch.tensor(rejected_attention_mask, dtype=torch.long, device=model.device)

        model.eval()
        with torch.no_grad():
            chosen_features = {"input_ids": chosen_input_ids, "attention_mask": chosen_attention_mask}
            rejected_features = {"input_ids": rejected_input_ids, "attention_mask": rejected_attention_mask}
            chosen_output = model(chosen_features)
            rejected_output = model(rejected_features)

        batch["chosen_embedding"] = [emb.cpu().tolist() for emb in chosen_output["sentence_embedding"]]
        batch["rejected_embedding"] = [emb.cpu().tolist() for emb in rejected_output["sentence_embedding"]]
        return batch

    @classmethod
    def post_process(cls, hf_dataset: HFDataset, dataset_config, tokenizer, model, data_path, rank_0_only=True) -> HFDataset:
        """
        Post-processes the dataset by computing embeddings for the response pairs.
        Args:
            hf_dataset: The dataset to process.
            dataset_config: The dataset configuration.
            tokenizer: The tokenizer.
            model: The model.
            data_path: The path to the data.
            rank_0_only: If True, only rank 0 will return the processed dataset.
        """
        logging.info(f"Precompute the embeddings for the response pairs: {dataset_config.dataset}. Length={len(hf_dataset)}")

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Shard the dataset so that each process gets a unique subset.
        if dist.is_initialized():
            hf_dataset = hf_dataset.shard(num_shards=world_size, index=rank)

        per_dp_cache_folder = Path(data_path).name
        # Make a cache folder that is unique per model and per process.
        dataset_map_cache_dir = Path(dataset_config.dataset_map_cache_dir) / per_dp_cache_folder / model.config._name_or_path.replace("/", "__")
        dataset_map_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
        cache_file_name = str(dataset_map_cache_dir / f"precompute_embedding_rank{rank}.arrow")

        # Adjust batch size: now each process only processes its shard.
        per_device_batch_size = dataset_config.post_process_per_device_batch_size
        batch_size_for_map = per_device_batch_size

        processed_dataset = hf_dataset.map(
            function=cls._precompute_response_embeddings,
            fn_kwargs={"tokenizer": tokenizer, "model": model},
            batched=True,
            batch_size=batch_size_for_map,
            num_proc=1,
            cache_file_name=cache_file_name,
            load_from_cache_file=False if dataset_config.force_refresh else True,
        )

        if world_size == 1:
            return processed_dataset

        # Save the processed shard to disk.
        shard_dir = dataset_map_cache_dir / f"shard_{rank}"
        processed_dataset.save_to_disk(str(shard_dir))
        # Wait for all processes to finish writing their shards.
        if dist.is_initialized():
            dist.barrier()

        if rank_0_only and rank != 0:
            return None
        else:
            shards = []
            for r in range(world_size):
                shard_path = dataset_map_cache_dir / f"shard_{r}"
                shard_ds = load_from_disk(str(shard_path))
                shards.append(shard_ds)
            merged_dataset = concatenate_datasets(shards)

        return merged_dataset

    def convert_to_features(self):
        logging.info(f"Transforming the dataset with prompt template: {self.dataset_config.dataset}. Length={len(self.hf_dataset)}")
        self.hf_dataset = self.hf_dataset.map(lambda batch: self._apply_prompt_template(batch, K=self.dataset_config.K_doc),
                                              batched=True,
                                              batch_size=1,  # Process one sample at a time
                                              num_proc=self.dataset_config.max_workers,
                                              cache_file_name=str(self.dataset_map_cache_dir / "apply_prompt_template.arrow"),
                                              load_from_cache_file=False if self.dataset_config.force_refresh else True,
                                              remove_columns=list(self.hf_dataset.features.keys())
                                              )

        logging.info(f"Transforming the dataset by grouping with group_id {self.dataset_config.group_column}: {self.dataset_config.dataset}. Length={len(self.hf_dataset)}")
        # Grouping step – ensures that each row now represents a full group
        self.hf_dataset = self._group_by_group_id(self.hf_dataset, "group_id")
        # Sample pair of rows from each group
        logging.info(f"Sampling pairs of rows from groups: {self.dataset_config.dataset}. Length={len(self.hf_dataset)}")
        # Note: better self.dataset_config.num_batches 2~4 times of self.dataset_config.max_workers to save cpu memory for large dataset
        batch_size_for_map = max(1, len(self.hf_dataset) // self.dataset_config.num_batches)
        self.hf_dataset = self.hf_dataset.map(lambda batch: self._sample_pairs_from_groups(batch, K=self.dataset_config.K_pair),
                                              batched=True,
                                              batch_size=batch_size_for_map,  # Process 'batch_size_for_map' groups at a time
                                              num_proc=self.dataset_config.max_workers,
                                              cache_file_name=str(self.dataset_map_cache_dir / "sample_pairs.arrow"),
                                              load_from_cache_file=False if self.dataset_config.force_refresh else True,
                                              remove_columns=list(self.hf_dataset.features.keys())
                                              )

        # take the name of the tokenizer as the cache folder, if it is a path, take the last part, if it is a model name, as is
        per_tokz_cache_folder = Path(self.tokenizer.name_or_path).name if Path(self.tokenizer.name_or_path).exists() else self.tokenizer.name_or_path
        per_tokz_cache_dir = self.dataset_map_cache_dir / per_tokz_cache_folder
        logging.info(f"Transforming the dataset with tokenization {per_tokz_cache_folder}: {self.dataset_config.dataset}. Length={len(self.hf_dataset)}")
        # create the folder is it doesn't exist
        if not per_tokz_cache_dir.exists():
            per_tokz_cache_dir.resolve().mkdir(parents=True, exist_ok=True)
        self.hf_dataset = self.hf_dataset.map(lambda sample: self._tokenize_add_label(sample, add_special_tokens=self.dataset_config.add_special_tokens),
                                              num_proc=self.dataset_config.max_workers,
                                              cache_file_name=str(per_tokz_cache_dir / "tokenize_add_label.arrow"),
                                              load_from_cache_file=False if self.dataset_config.force_refresh else True,
                                              remove_columns=list(self.hf_dataset.features.keys())
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
        currPath = Path.cwd() / "controlllm" / "data" / "mock_data"
        logging.info(f"Loading dataset {currPath}")
        data_path = str(currPath)
    dataset = SemanticSearch(
        dataset_config,
        tokenizer=tokenizer,
        data_path=data_path
    )

    return dataset.hf_dataset


def postprocess_dataset(hf_dataset, dataset_config, tokenizer, model, data_path) -> HFDataset:
    """
    Post-processes the dataset by computing embeddings for the response pairs.
    """
    return SemanticSearch.post_process(hf_dataset, dataset_config, tokenizer, model, data_path)


class SemanticSearchDataCollator:
    def __init__(self, processor: PreTrainedTokenizer):
        """
        For LLM, the processor is typically the tokenizer itself.
        """
        self.processor = processor

    def pad_pair(self, features: List[Dict], input_key: str, mask_key: str) -> Dict[str, torch.Tensor]:
        """
        Pads a pair of input_ids and attention_mask to the same length.
        """
        # Extract the relevant fields and rename them to "input_ids" and "attention_mask"
        inputs = [{"input_ids": f[input_key], "attention_mask": f[mask_key]} for f in features]

        # Temporarily set model_input_names to include the relevant keys
        original_model_input_names = self.processor.model_input_names
        self.processor.model_input_names = ["input_ids", "attention_mask"]

        # Let the tokenizer pad the fields
        padded = pad_without_fast_tokenizer_warning(
            self.processor,
            inputs,
            padding=True,
            return_tensors="pt",
        )

        # Restore the original model_input_names
        self.processor.model_input_names = original_model_input_names

        # Rename the keys back to their original names
        padded = {input_key: padded["input_ids"], mask_key: padded["attention_mask"]}

        return padded

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Input is a list of samples from the dataset:
        {
            "prompt_input_ids": ...,
            "chosen_input_ids": ...,
            "rejected_input_ids": ...,
            "prompt_attention_mask": ...,
            "chosen_attention_mask": ...,
            "rejected_attention_mask": ...,
            "label": sample["margin"],
            "group_id": sample["group_id"],
            "score_chosen": sample["score_chosen"],
            "score_reject": sample["score_reject"]

            "chosen_embedding": ...,  # precomputed embeddings when enable_post_process=True configured in ./configs/datasets.py
            "rejected_embedding": ...,  # precomputed embeddings when enable_post_process=True configured in ./configs/datasets.py
        }
        Output is a dictionary of tensors, where all input_ids and attention_mask
        are padded by self.processor.padding_side ("left" or "right").
        """
        if not features:
            return {}

        # Determine if we have a "label" key
        label_key = "label" if "label" in features[0] else None
        labels = [f[label_key] for f in features] if label_key else None

        # Find all unique prefixes for input_ids and attention_mask pairs
        prefixes = set()
        for key in features[0].keys():
            if "input_ids" in key:
                prefix = key.replace("_input_ids", "")
                prefixes.add(prefix)
            elif key == "input_ids":
                prefixes.add("")

        # Pad each pair of input_ids and attention_mask
        batch = {}
        for prefix in prefixes:
            if prefix:
                input_key = f"{prefix}_input_ids"
                mask_key = f"{prefix}_attention_mask"
            else:
                input_key = "input_ids"
                mask_key = "attention_mask"
            padded = self.pad_pair(features, input_key, mask_key)
            batch.update(padded)

        # Convert labels to torch tensors if present
        if labels is not None:
            if isinstance(labels[0], float):
                batch["label"] = torch.tensor(labels, dtype=torch.float)
            elif isinstance(labels[0], int):
                batch["label"] = torch.tensor(labels, dtype=torch.long)
            else:
                labels = [float(label) for label in labels]
                batch["label"] = torch.tensor(labels, dtype=torch.float)

        # Handle any other keys that are not in input_id_keys or attention_mask_keys
        # (like group_id, score_chosen, etc.)
        for key in features[0].keys():
            # skip label_key and already-padded fields
            if key == label_key or key in batch:
                continue

            vals = [f[key] for f in features]
            # Convert numeric data to tensors, leave strings as is
            if isinstance(vals[0], float):
                batch[key] = torch.tensor(vals, dtype=torch.float)
            elif isinstance(vals[0], int):
                batch[key] = torch.tensor(vals, dtype=torch.long)
            elif isinstance(vals[0], list):
                # e.g., list of lists (embeddings)
                batch[key] = torch.tensor(vals, dtype=torch.float)
            else:
                # e.g., str or array
                batch[key] = vals

        return batch


def get_data_collator(processor):
    return SemanticSearchDataCollator(processor)
