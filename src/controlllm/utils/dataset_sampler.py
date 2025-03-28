import math
import random
import logging
import numpy as np
from tqdm import tqdm
from itertools import islice
from typing import List, Dict, Optional, TypeVar, Iterator

import torch
from torch.utils.data import Dataset, BatchSampler, DistributedSampler

from controlllm.configs.training import TrainConfigCommon


def get_memory_available() -> int:
    """
    Returns the available memory in bytes on the GPU."""
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    return (total_memory - reserved_memory)  # Returns available memory in bytes


def dynamic_token_limit(train_config: TrainConfigCommon, batch_size: int, debug:bool=False) -> int:
    """
    Calculates the maximum number of tokens per batch based on available GPU memory.

    Parameters:
        train_config (TrainConfigCommon): Configuration containing memory settings.
        batch_size (int): Number of items in each batch.

    Returns:
        int: Maximum tokens per batch that can fit in the available GPU memory.
    """
    # Based on 80GB GPU memory: 8192 * 8 = 65,536 tokens fit in the available memory
    reference_tokens = batch_size * train_config.context_length  # adjust this depending on model size

    if debug:  # if debug is enabled, compute the memory_per_token to understand the best practice in given infra/model_training settings
        # Get available GPU memory after the model has been loaded
        memory_available = get_memory_available()
        if train_config.memory_per_token is None or train_config.memory_per_token == -1:
            if batch_size == 0 or train_config.context_length == 0:
                raise ValueError("batch_size and train_config.context_length must be greater than 0")

            memory_per_token = memory_available / reference_tokens  # This gives the memory used per token in bytes
            logging.debug(f"{train_config.memory_per_token=}. Calculated {memory_per_token=} by: {batch_size=} * {train_config.context_length=} = {reference_tokens}")
        else:
            memory_per_token = train_config.memory_per_token
            logging.debug(f"Using user-defined memory_per_token: {train_config.memory_per_token=} bytes")

        # Calculate the max tokens per batch that can fit in the currently available memory
        max_tokens_per_batch = int(memory_available / memory_per_token)
        logging.debug(f"Calculated dynamic max_tokens_per_batch: {max_tokens_per_batch} using {memory_available=} / {memory_per_token=} bytes")
    else:
        # train_config.memory_per_token == -1 means to set max_tokens_per_batch = batch_size * train_config.context_length
        logging.debug(f"Setting max_tokens_per_batch = {batch_size=} * {train_config.context_length=} => {reference_tokens=} tokens as it is configured as -1.")
        max_tokens_per_batch = batch_size * train_config.context_length

    return max_tokens_per_batch


def create_batches(train_config: TrainConfigCommon, lengths: List[int], indices: List[int], batch_size: int, debug: bool=False) -> Dict[int, List[List[int]]]:
    """
    Create batches based on lengths of the data points, returns dict of key being phase number and value being list of dynamic or fix batches
    Note indices is the original indices while lengths is the corresponsding length in the dataset, each batch is a list of original indices of the data points

    This func is called in multi-threading which can result in excessive logging, so logging is only enabled in debugging mode
    """

    # if debug is enabled, set the logging level to debug, revert it back to the original level after the function is done
    if debug:
        logging_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.DEBUG)
    tqdm_disabled = logging.getLogger().getEffectiveLevel() > logging.DEBUG

    # Sort data by length for better batching
    sorted_idx = np.argsort(lengths, kind='mergesort')
    sorted_indices = [indices[i] for i in sorted_idx]  # Remap sorted indices to the original indices
    sorted_lengths = [lengths[i] for i in sorted_idx]  # Keep lengths aligned with sorted_indices

    phase_batches = {}

    # Create batches based on lengths
    if not train_config.dynamic_batch_size:  # fixed batch size
        # drop last incomplete batch only if not using dynamic batch size
        logging.debug("Dropping last incomplete batch")
        if train_config.drop_last:
            sorted_indices = sorted_indices[:len(sorted_indices) // batch_size * batch_size]
            sorted_lengths = sorted_lengths[:len(sorted_lengths) // batch_size * batch_size]

        if train_config.handle_long_sequences:  # drop data point longer than threshold, this will avoid OOM error
            original_len = len(sorted_indices)
            sorted_indices = [sorted_indices[i] for i in range(len(sorted_lengths)) if sorted_lengths[i] <= train_config.long_sequence_threshold]
            num_long_sequences = original_len - len(sorted_indices)
            if num_long_sequences > 0:
                logging.warning(f"Dropped {num_long_sequences} long sequences with length > {train_config.long_sequence_threshold}")

        short_batches = [sorted_indices[i:i+batch_size] for i in tqdm(range(0, len(sorted_indices), batch_size), desc="Creating batches", disable=tqdm_disabled) if sorted_lengths[i] <= train_config.context_length]
        logging.info(f"Created {len(short_batches)} batches for sequences shorter than or equal to {train_config.context_length} with fixed batch size {batch_size}")
        if short_batches:
            phase_batches.setdefault(0, []).extend(short_batches)

        if train_config.handle_long_sequences:
            # collect all that is length is > train_config.context_length, reduce the batch size to quadratic factor to avoid OOM
            quad_factor = (train_config.long_sequence_threshold // train_config.context_length) ** 2  # assume quadratic relationship between memory usage and sequence length
            reduced_batch_size = max(1, batch_size // quad_factor)
            long_batches = [sorted_indices[i:i+reduced_batch_size] for i in tqdm(range(0, len(sorted_indices), reduced_batch_size), desc="Creating batches", disable=tqdm_disabled) if sorted_lengths[i] > train_config.context_length]
            logging.info(f"Created {len(long_batches)} batches for sequences longer than {train_config.context_length} with fixed batch size {reduced_batch_size}")
            if long_batches:
                phase_batches.setdefault(1, []).extend(long_batches)
        else:
            long_batches = [sorted_indices[i:i+batch_size] for i in tqdm(range(0, len(sorted_indices), batch_size), desc="Creating batches", disable=tqdm_disabled) if sorted_lengths[i] > train_config.context_length]
            logging.info(f"Created {len(long_batches)} batches for sequences longer than {train_config.context_length} with fixed batch size {batch_size}. You may want to enable handle_long_sequences to avoid OOM errors.")
            if long_batches:
                phase_batches.setdefault(0, []).extend(long_batches)
    else:
        # Optimizes GPU utilization by adapting the batch size based on available memory
        if train_config.max_tokens_per_batch == -1:
            max_tokens_per_batch = dynamic_token_limit(train_config=train_config, batch_size=batch_size, debug=debug)
            logging.debug(f"{train_config.max_tokens_per_batch=} so dynamically computed and setting max_tokens_per_batch to {max_tokens_per_batch}")
        else:
            max_tokens_per_batch = train_config.max_tokens_per_batch
            logging.debug(f"Setting max_tokens_per_batch to {train_config.max_tokens_per_batch=}")

        current_batch = []
        current_token_count = 0

        if train_config.curriculum_learning:
            phase_length = (max(lengths) - min(lengths)) // train_config.curriculum_phases
            logging.debug(f"Curriculum learning enabled with {train_config.curriculum_phases} phases and {phase_length=}")
        else:
            logging.debug("Curriculum learning disabled, we will use a single phase which means shuffling all the data during training")
            phase_length = float('inf')
        current_phase = 0

        for idx, item_length in tqdm(zip(sorted_indices, sorted_lengths), total=len(sorted_indices), desc="Creating batches", disable=tqdm_disabled):
            if item_length > max_tokens_per_batch:  # TODO: don't add this item to the batch, log a warning
                logging.warning(f"Item length {item_length} at index {idx} is greater than max_tokens_per_batch {max_tokens_per_batch}. This will likely cause OOM errors. Dropping this item.")
                continue

            # Curriculum phase advancement
            if train_config.curriculum_learning and item_length > min(lengths) + (current_phase + 1) * phase_length:
                logging.debug(f"Advancing to phase {current_phase + 1} due to item length {item_length} > {(current_phase + 1) * phase_length}")
                if current_batch:  # Close the current batch before advancing phase
                    phase_batches.setdefault(current_phase, []).append(current_batch)

                current_phase += 1
                current_batch = []
                current_token_count = 0

            # Then, check if a new batch should be started due to capacity constraints
            # If the current batch is full in terms of max_tokens_per_batch or the item is too long, start a new batch before that item is added
            if (current_token_count + item_length > max_tokens_per_batch and len(current_batch) > 0) or \
            (train_config.handle_long_sequences and item_length > train_config.long_sequence_threshold):
                if current_batch:
                    phase_batches.setdefault(current_phase, []).append(current_batch)

                current_batch = []
                current_token_count = 0

            current_batch.append(idx)
            current_token_count += item_length

        # Append the last batch
        if current_batch:
            phase_batches.setdefault(current_phase, []).append(current_batch)

    # if debug is enabled, set the logging level to debug, revert it back to the original level after the function is done
    if debug:
        logging.getLogger().setLevel(logging_level)

    return phase_batches


class LengthBasedBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, train_config: TrainConfigCommon, batch_size: int, shuffle: bool=True, seed: int=0) -> None:
        """
        Creates batches of similar lengths by sorting the data points by length and then creating batches of similar lengths."""
        self.seed = seed
        self.epoch = 0
        self.starting_step = 0  # The step from which to resume training, to be updated on the fly by train_utils.py
        self.current_step = 0  # To track steps within the iteration
        self._starting_step_applied = False  # Track whether starting_step has been applied for this epoch
        self.train_config = train_config
        self.batch_size = batch_size
        self.shuffle = shuffle

        # if dataset already has lengths and phase_batches preprocessed, use them, otherwise compute them
        msg_long_running_process = "Padding: computing lengths of all data points to get batches of similar lengths by mergesort."
        if dataset and hasattr(dataset, 'lengths') and dataset.lengths:
            self.lengths = dataset.lengths
        elif dataset and isinstance(next(iter(dataset)), dict):
            logging.info(msg_long_running_process)
            batch = next(iter(dataset))
            input_ids_keys = [key for key in batch.keys() if "input_ids" in key]

            if not input_ids_keys:
                raise ValueError("No 'input_ids' keys found in the dataset batch.")

            total_lengths = [0] * len(batch[input_ids_keys[0]])

            for key in input_ids_keys:
                lengths = [len(d[key]) for d in tqdm(dataset, desc=f"Computing data lengths of {key}")]
                total_lengths = [sum(x) for x in zip(total_lengths, lengths)]

            self.lengths = total_lengths
        else:
            logging.info(msg_long_running_process)
            self.lengths = [len(d) for d in tqdm(dataset, desc="Computing data lengths")]

        if hasattr(dataset, 'phase_batches') and dataset.phase_batches:
            self.phase_batches = dataset.phase_batches

            # handle case when data preprocessing is done with curriculum learning enabled, but the training config has it disabled in later runs
            if not train_config.curriculum_learning and len(self.phase_batches) > 1:
                logging.warning("Curriculum learning is disabled, but phase_batches is provided with more than one phase. We will merge them into a single phase.")
                self.phase_batches = {0: [batch for phase in self.phase_batches.values() for batch in phase]}
        else:
            self.phase_batches = create_batches(train_config=train_config, lengths=self.lengths, indices=list(range(len(self.lengths))), batch_size=batch_size, debug=train_config.debug)

    def __iter__(self):
        self.current_step = 0  # Reset current_step for every new iteration (epoch)
        assert self.starting_step < len(self), f"starting_step {self.starting_step} is greater than or equal to the total number of batches {len(self)}."

        # yield the last batch with the longest sequence in each phase first to fail and fail fast with OOM
        try:
            for phase in sorted(self.phase_batches.keys()):
                logging.info(f"Pressure test: yielding the last batch with the longest sequence in phase {phase}")
                if self.current_step >= self.starting_step and not self._starting_step_applied:
                    self._starting_step_applied = True  # Mark that starting_step has been applied

                if self._starting_step_applied:
                    yield self.phase_batches[phase][-1]

                self.current_step += 1
        except (ValueError, IndexError):
            pass  # Handle empty self.phase_batches or empty last list case

        if not self.train_config.curriculum_learning:  # merge all phases into a single phase
            self.phase_batches = {0: [batch for phase in self.phase_batches.values() for batch in phase]}

        for phase in sorted(self.phase_batches.keys()):  # Process each phase in order, if curriculum learning is disabled, this will be a single phase
            phase_data = self.phase_batches[phase]
            if self.shuffle:
                # putting this in __iter__ not in __init__ to ensure shuffling between epochs
                random.seed(self.seed + self.epoch)
                random.shuffle(phase_data)  # Shuffle within the phase
            for batch in phase_data:
                if self.current_step >= self.starting_step and not self._starting_step_applied:
                    self._starting_step_applied = True  # Mark that starting_step has been applied

                if batch and self._starting_step_applied:  # Skip batches until starting_step
                    yield batch

                self.current_step += 1  # Increment step for every batch

        self.epoch += 1  # Increment the epoch after all batches are yielded to enable different shuffling between epochs

    def __len__(self):
        total_batches = len(self.phase_batches)  # Start with number of phases to account for the additional yield of last batch with the longest sequence
        for phase in self.phase_batches:
            total_batches += len(self.phase_batches[phase])

        return total_batches

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def set_starting_step(self, starting_step: int) -> None:
        r"""
        Set the starting step for this sampler.

        Args:
            starting_step (int): Starting step number.
        """
        if starting_step >= len(self):
            raise ValueError(f"starting_step {starting_step} is greater than or equal to the total number of samples {len(self.dataset)}.")
        self.starting_step = starting_step
        self._starting_step_applied = False  # Ensure starting_step is applied on the next iteration


class DistributedLengthBasedBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, train_config: TrainConfigCommon, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        """
        Creates batches of similar lengths by sorting the data points by length and then creating batches of similar lengths. This sampler is used in distributed training."""
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(dataset=dataset, train_config=train_config, batch_size=batch_size, shuffle=shuffle, seed=seed)
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size  # transformers trainer uses this to compute even batch

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        # Ensures either all active replicas (up to max_length) get batches, or none do. At the cost that some batches may be skipped.
        return len(self.batch_sampler) // self.num_replicas


T_co = TypeVar('T_co', covariant=True)

class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, batch_size: int, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, starting_step=0) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.starting_step = starting_step
        self._starting_step_applied = False  # Track whether starting_step has been applied for this epoch
        self.batch_size = batch_size  # transformers trainer uses this to compute even batch

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # Handle padding globally, but only apply it to the replica's indices after splitting
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Subsample for the current rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Handle starting_step logic: Skip the first `starting_step` samples only once
        if self.starting_step > 0 and not self._starting_step_applied:
            indices = [idx for idx in indices if idx < len(self.dataset)]  # Exclude padded indices
            indices = indices[self.starting_step:]  # Skip until `starting_step`
            self._starting_step_applied = True  # Mark starting_step as applied for this epoch

        # Reapply padding logic after skipping
        if len(indices) < self.num_samples:
            padding_size = self.num_samples - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def set_starting_step(self, starting_step: int) -> None:
        r"""
        Set the starting step for this sampler.

        Args:
            starting_step (int): Starting step number.
        """
        if starting_step >= len(self.dataset):
            raise ValueError(f"starting_step {starting_step} is greater than or equal to the total number of samples {len(self.dataset)}.")
        self.starting_step = starting_step
        self._starting_step_applied = False  # Ensure starting_step is applied on the next iteration
