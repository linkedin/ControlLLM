import os
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import itertools
import json
import ast
import random
from typing import List
import fastavro
import fsspec
import subprocess
from pathlib import Path
import pyarrow as pa
from datasets import Dataset, Features, Value, Sequence, concatenate_datasets, ClassLabel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast as Tokenizer
from controlllm.model.llama3 import Dialog
from controlllm.configs.datasets import AbstractDataset as DatasetConfig
from controlllm.model.llama3.tokenizer import Message


def load_avro_dirs_as_dataset(avro_dir: str, avro_dataset_cache_dir: str, max_workers: int=8) -> Dataset:
    """
    Helper function loading all the avro data under a directory as a dataset
    Uses multiprocessing to shard avro loading of different files across different processes
    """

    # check if the avro_dir(a hdfs path) is empty, if so, try to load it from cache, if not cached, return None
    if not list(fsspec.open_files(os.path.join(avro_dir, "*.avro"))):
        logging.warning(f"Avro dir {avro_dir} is empty. Try to load from cache: {avro_dataset_cache_dir}")
        return load_dataset_from_cache(cache_dir=avro_dataset_cache_dir)

    # i/o bound, during this wait for external resource, the GIL can be released, allowing other threads to run
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        datasets = list(executor.map(
            partial(_load_avro_dirs_as_dataset_for_rank, avro_dir, max_workers, avro_dataset_cache_dir),
            range(max_workers),
        ))
    return concatenate_datasets(list(filter(lambda dataset: dataset is not None, datasets)))


def _load_avro_dirs_as_dataset_for_rank(
    avro_dir: str, max_workers: int, avro_dataset_cache_dir: str, rank: int
) -> Dataset:
    """
    Helper to load avro dirs for multiprocessing within a rank. Each rank only loads a fraction of the
    avro files in the dir using mod function.
    """
    avro_files_for_rank = list()
    # Need to call fsspec within multiprocessing and not in main thread. There is an odd bug where
    # hitting HDFS client to open avro files outside of main thread leads to deadlock.
    # See https://github.com/crs4/pydoop/issues/311 for more details.
    for i, avro_file in enumerate(fsspec.open_files(os.path.join(avro_dir, "*.avro"))):
        if i % max_workers == rank:
            avro_files_for_rank.append(avro_file)
    logging.info(f"Rank {rank} loading {avro_files_for_rank}")

    def generator():
        for avro_file in avro_files_for_rank:
            with avro_file as f:
                reader = fastavro.reader(f)
                # there might be error in particular row of the avro file, so we need to catch the exception and continue
                while True:
                    try:
                        r = next(reader)
                        yield r
                    except StopIteration:
                        break
                    except Exception as e:
                        logging.error(f"Error processing one record in {avro_file}: {e}. Continue...")

    if avro_files_for_rank:
        dataset = Dataset.from_generator(generator, cache_dir=avro_dataset_cache_dir)
        return dataset
    else:
        return None


def load_dataset_from_cache(cache_dir: str, save_dir: str=None, key: str=None) -> Dataset:
    """
    Load huggingface dataset from cache
    """

    # if it is already saved in save_dir with .arrow files there, load it from the save_dir
    if save_dir is None:
        save_dir = os.path.join(cache_dir, "cache")

    if os.path.exists(save_dir) and list(fsspec.open_files(f"{save_dir}/*.arrow")):
        logging.info(f"Loading dataset from the saved cache folder: {save_dir}")
        dataset = Dataset.load_from_disk(save_dir)
        if key:
            def remove_duplicates_efficiently(dataset):
                # Step 1: Sort the dataset by key
                sorted_dataset = dataset.sort(key)
                # Step 2: Add a column to flag the first occurrence of each unique key
                def flag_unique(batch):
                    # Create an array that flags changes in key
                    unique_flags = [1] + [int(batch[key][i] != batch[key][i - 1]) for i in range(1, len(batch[key]))]
                    return {'is_unique': unique_flags}
                # Apply this function batch-wise
                sorted_dataset = sorted_dataset.map(flag_unique, batched=True, batch_size=10_000)
                # Step 3: Filter to keep only rows where 'is_unique' is 1 (true)
                unique_dataset = sorted_dataset.filter(lambda example: example['is_unique'] == 1)
                # Remove the 'is_unique' column if it's no longer needed
                unique_dataset = unique_dataset.remove_columns('is_unique')
                return unique_dataset

            dataset = remove_duplicates_efficiently(dataset)

        return dataset

    def extract_features_from_json(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
            features_info = data['features']
            features = {}
            for feature_name, attrs in features_info.items():
                if isinstance(attrs, list):  # Handling nested structures as in Format B
                    nested_features = {}
                    for item in attrs:
                        for key, val in item.items():
                            dtype = val['dtype']
                            if val['_type'] == 'Value':
                                nested_features[key] = Value(dtype)
                            # Add other types as needed
                    features[feature_name] = Sequence(nested_features)  # Use Sequence for nested features
                else:
                    dtype = attrs['dtype']
                    if attrs['_type'] == 'Value':
                        features[feature_name] = Value(dtype)
                    elif attrs['_type'] == 'ClassLabel':
                        features[feature_name] = ClassLabel(num_classes=attrs['num_classes'])
                    # Add other types as needed
            return Features(features)

    def load_and_concatenate_datasets(base_path):
        """
        Load and concatenate all datasets in the base_path
        """
        dataset_list = []
        features_defined = False

        for folder in os.listdir(base_path):
            folder_path = Path(base_path) / folder / "0.0.0"
            if os.path.isdir(folder_path):
                json_file = Path(folder_path) / 'dataset_info.json'
                if os.path.exists(json_file) and not features_defined:
                    # Extract features from the first found dataset_info.json
                    features = extract_features_from_json(json_file)
                    features_defined = True

                for file in os.listdir(folder_path):
                    if file.endswith(".arrow"):
                        file_path = Path(folder_path) / file
                        # Load the Arrow Table using RecordBatchStreamReader
                        with open(file_path, 'rb') as f:
                            reader = pa.ipc.open_stream(f)
                            for batch in reader:
                                try:
                                    dataset = Dataset.from_pandas(batch.to_pandas(), features=features)
                                except Exception as e:
                                    logging.error(f"Error loading Arrow file {file_path}: {e}")
                                    continue
                                dataset_list.append(dataset)

        # Concatenate all datasets
        if dataset_list:
            concatenated_dataset = concatenate_datasets(dataset_list)
            return concatenated_dataset
        else:
            return None

    base_directory = Path(cache_dir) / "generator"
    final_dataset = load_and_concatenate_datasets(base_directory) if os.path.exists(base_directory) else None

    if final_dataset is not None:
        final_dataset.save_to_disk(save_dir)
        logging.info(f"Dataset cached in {cache_dir} successfully saved to {save_dir}.")
    else:
        logging.info(f"No datasets were loaded likely due to it has not been cached in {cache_dir}.")

    return final_dataset


def load_arrow_dirs_as_dataset(arrow_dir: str, arrow_dataset_cache_dir: str, dataset_config: DatasetConfig) -> Dataset:
    """
    Helper function loading the arrow data in hdfs that has been saved by hugingface dataset.save_to_disk
    """
    # Check if the arrow_dir(a hdfs path) is empty, if so, raise error, else load it by Dataset.load_from_disk enabled by hdfs
    if not list(fsspec.open_files(f"{arrow_dir}/*.arrow")):
        raise ValueError(f"Arrow dir {arrow_dir} is empty.")

    # Check if the arrow_dir is a hdfs path(e.g. hdfs://ltx1-holdem/jobs/controlllm/...) of shared nfs file path(e.g. /shared/public/data/controlllm/datasets/...)
    if arrow_dir.startswith("hdfs"):
        # Create the local directory if it doesn't exist
        os.makedirs(arrow_dataset_cache_dir, exist_ok=True)

        # Check if there is files in the cache directory, if so, return the dataset
        if not os.listdir(arrow_dataset_cache_dir) or dataset_config.force_refresh:
            # Download the dataset from HDFS to the local file system
            logging.info(f"Downloading dataset from {arrow_dir} to {arrow_dataset_cache_dir}...")
            arrow_dir = arrow_dir + '/*'  # HDFS source path with wildcard

            # Using shell=True to allow wildcard expansion by the shell
            process = subprocess.run(f'hdfs dfs -get -f {arrow_dir} {arrow_dataset_cache_dir}', shell=True, check=True)
            process.check_returncode()  # check if the process was successful    
            logging.info(f"Downloaded dataset from {arrow_dir} to {arrow_dataset_cache_dir} successfully.")
        else:
            logging.info(f"Dataset already exists in the cache directory: {arrow_dataset_cache_dir} and force_refresh is {dataset_config.force_refresh}. Skipping download.")
    else:
        arrow_dataset_cache_dir = arrow_dir

    # Load the dataset from the local file system
    logging.info(f"Loading dataset from {arrow_dataset_cache_dir}...")
    dataset = Dataset.load_from_disk(arrow_dataset_cache_dir)
    logging.info(f"Loaded dataset from {arrow_dataset_cache_dir} successfully. Length: {len(dataset)}.")

    return dataset


def sample_dataset(hf_dataset: Dataset, dataset_config: DatasetConfig, split: str) -> Dataset:
    """
    Sample a subset of the dataset for each huggface dataset in [dataset.hf_dataset for dataset in self.dataset_train] according to dataset.dataset_config.sample_size
    dataset.dataset_config.sample_size is the number of samples to sample from the dataset if >1 or percentage of the dataset to sample if <=1
    """
    if float(dataset_config.sample_size) == 1.0:
        logging.info(f"Skipping sampling for dataset: {dataset_config.dataset} - {split} as sample_size is 1.0 which means 100% of the dataset.")
    else:
        # random sampling in place
        num_samples = int(dataset_config.sample_size) if dataset_config.sample_size > 1 and isinstance(dataset_config.sample_size, int) else max(int(len(hf_dataset) * dataset_config.sample_size), 1)
        logging.info(f"Sampling {num_samples} samples from the dataset: {dataset_config.dataset} - {split}. Which is {dataset_config.sample_size} of the dataset size {len(hf_dataset)}")
        if num_samples > len(hf_dataset):
            # Upsampling: sampling with replacement
            random_indices = random.choices(range(len(hf_dataset)), k=num_samples)
        else:
            # Downsampling: sampling without replacement
            random_indices = random.sample(range(len(hf_dataset)), num_samples)
        hf_dataset = hf_dataset.select(random_indices)

    return hf_dataset


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"

U_INST, S_INST, A_INST = "<|user|>", "<|system|>", "<|assistant|>"


class Llama3_ChatFormat:
    """
    Chat format for llama 3
    """
    def __init__(self, tokenizer: Tokenizer, dataset_config: DatasetConfig):
        self.tokenizer = tokenizer
        self.dataset_config = dataset_config

    def encode_bot(self):
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    def encode_header(self, message: Message) -> List[int]:
        tokens = []
        tokens.extend(self.tokenizer.encode("<|start_header_id|>", add_special_tokens=False))
        tokens.extend(self.tokenizer.encode(message["role"], add_special_tokens=False))
        tokens.extend(self.tokenizer.encode("<|end_header_id|>", add_special_tokens=False))
        tokens.extend(self.tokenizer.encode("\n\n", add_special_tokens=False))
        return tokens

    def encode_message(self, message: Message) -> List[int]:
        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), add_special_tokens=False, truncation=self.dataset_config.truncation, max_length=self.dataset_config.max_length)
        )
        tokens.extend(self.tokenizer.encode("<|eot_id|>", add_special_tokens=False))
        return tokens

    def encode_dialog_prompt(self, dialog: Dialog) -> List[int]:
        tokens = []
        tokens.extend(self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False))
        for message in dialog:
            tokens.extend(self.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        tokens.extend(self.encode_header({"role": "assistant", "content": ""}))
        return tokens


def tokenize_dialog(dialog, tokenizer, dataset_config, inference=False):
    """
    Tokenize dialog(list of messages) into features for training and inference
    """
    if isinstance(dialog, str):
        try:
            # Parse the modified string as JSON
            dialog = ast.literal_eval(dialog)
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, try json.loads
            try:
                dialog = json.loads(dialog)
            except ValueError as e:
                print("Validation error:", e)

    if dataset_config.chat_template.upper() == "LLAMA3":
        """ example dialog after applying chat template:

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
        You are a helpful AI assistant for travel tips and recommendations<|eot_id|>

        <|start_header_id|>user<|end_header_id|>\n\n
        What is France's capital?<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>\n\n
        Bonjour! The capital of France is Paris!<|eot_id|>

        <|start_header_id|>user<|end_header_id|>\n\n
        What can I do there?<|eot_id|>

        <|start_header_id|>assistant<|end_header_id|>\n\n
        Paris, the City of Light...<|eot_id|>

        <|start_header_id|>user<|end_header_id|>\n\n
        Give me a detailed list of the attractions I should visit.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

        <|start_header_id|>assistant<|end_header_id|>\n\n --> inference == True
        """

        # chat_template: chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}", response_template_with_context='<|start_header_id|>assistant<|end_header_id|>\n\n'
        chat_format = Llama3_ChatFormat(tokenizer, dataset_config)
        tokens = []
        tokens.extend(chat_format.encode_bot())
        system_prompt = None
        if dialog[0]['role'] == 'system':
            tokens.extend(chat_format.encode_message(dialog[0]))
            dialog = dialog[1:]
        prompt_tokens = [tokens + chat_format.encode_message(prompt) if i==0 else chat_format.encode_message(prompt) for i, prompt in enumerate(dialog[::2])]
        answer_tokens = [chat_format.encode_message(answer) for answer in dialog[1::2]]

        if dataset_config.pretrain:
            answer_tokens[-1].extend(tokenizer.encode("<|end_of_text|>", add_special_tokens=False))

        if inference:
            if len(prompt_tokens) != len(answer_tokens) + 1:
                raise ValueError("Prompt length should be one more than answer length for inference. e.g. User: Assistant: User: ")
            generation_prompt = chat_format.encode_header({"role": "assistant", "content": ""})
            answer_tokens.append(generation_prompt)

    elif dataset_config.chat_template.upper() == "LLAMA2":  # this is for llama 2
        """ example dialog after applying chat template:

        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>

        <s>[INST] {{ user_message_2 }} [/INST]  {{ model_answer_1 }} </s>

        <s>[INST] {{ user_message_2 }} [/INST] --> inference == True
        """
        # chat_template: {% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
        system_prompt = None
        if dialog[0]['role'] == 'system':
            system_message = dialog[0]['content']
            dialog = dialog[1:]
            system_prompt = f"{B_SYS}\n{system_message}\n{E_SYS}\n\n"
        encode = partial(tokenizer.encode, add_special_tokens=False, truncation=dataset_config.truncation, max_length=dataset_config.max_length)
        prompt_tokens = [[tokenizer.bos_token_id] + encode(f"{B_INST} {system_prompt if system_prompt is not None and i == 0 else ''} {(prompt['content']).strip()} {E_INST}") for i, prompt in enumerate(dialog[::2])]
        answer_tokens = [encode(f"{answer['content'].strip()}") + [tokenizer.eos_token_id] for answer in dialog[1::2]]

        if inference:
            if len(prompt_tokens) != len(answer_tokens) + 1:
                raise ValueError("Prompt length should be one more than answer length for inference. e.g. User: Assistant: User: ")
            answer_tokens.append([])

    elif dataset_config.chat_template.upper() == "MISTRAL":
        """ example dialog after applying chat template:
        Note that mistral/mixtral(e.g. Mistral-7B-Instruct-v0.2)'s default prompt template does not support system prompt, customize it according to llama2 formate
        reference: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/discussions/52

        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>

        {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>

        <s>[INST] {{ user_message_2 }} [/INST]{{ model_answer_1 }}</s>

        <s>[INST] {{ user_message_3 }} [/INST] --> inference == True
        """
        # chat_template: {% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}
        system_prompt = None
        if dialog[0]['role'] == 'system':
            system_message = dialog[0]['content']
            dialog = dialog[1:]
            system_prompt = f"{B_SYS}\n{system_message}\n{E_SYS}\n\n"
        encode = partial(tokenizer.encode, add_special_tokens=False, truncation=dataset_config.truncation, max_length=dataset_config.max_length)
        prompt_tokens = [[tokenizer.bos_token_id] + encode(f"{B_INST} {system_prompt if system_prompt is not None and i == 0 else ''} {(prompt['content']).strip()} {E_INST}") for i, prompt in enumerate(dialog[::2])]
        answer_tokens = [encode(f"{answer['content'].strip()}") + [tokenizer.eos_token_id] for answer in dialog[1::2]]

        if inference:
            if len(prompt_tokens) != len(answer_tokens) + 1:
                raise ValueError("Prompt length should be one more than answer length for inference. e.g. User: Assistant: User: ")
            answer_tokens.append([])

    elif dataset_config.chat_template.upper() == "AUTO":
        if inference:
            input_ids = tokenizer.apply_chat_template(dialog, tokenizer=True, add_generation_prompt=True)
            return {
                "input_ids": input_ids,
                "labels": input_ids,
                "attention_mask": [1] * len(input_ids["input_ids"])
            }
        else:
            # this is for auto chat template, which is a generic chat template that can be used for any model

            # commented out code below is for template without `{% generation %}` keyword, need to customize it according to the model's chat template
            # system_prompt = ""
            # if dialog[0]['role'] == 'system':
            #     system_message = dialog[0]['content']
            #     dialog = dialog[1:]
            #     system_prompt = tokenizer.apply_chat_template(dialog[0], tokenize=False, add_generation_prompt=False)

            # def apply_chat_template(message):
            #     message_prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
            #     # left split by the first eos_token, and remove the system prompt
            #     return message_prompt.split(tokenizer.eos_token, 1)[1]

            # encode = partial(tokenizer.encode, add_special_tokens=False, truncation=dataset_config.truncation, max_length=dataset_config.max_length)
            # prompt_tokens = [encode(system_prompt if i == 0 else "" + apply_chat_template(prompt)) for i,  prompt in enumerate(dialog[::2])]
            # answer_tokens = [encode(apply_chat_template(answer)) for answer in dialog[1::2]]

            # This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
            output = tokenizer.apply_chat_template(dialog, tokenizer=True, add_generation_prompt=False, return_assistant_tokens_mask=True, return_dict=True)
            input_ids = output["input_ids"]
            assistant_masks = output["assistant_masks"]
            # For tokens generated by the assistant, the mask will contain 1. For user and system tokens, the mask will contain 0.
            labels = [input_ids[i] if assistant_masks[i] else -100 for i in range(len(input_ids))]
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids["input_ids"])
            }

    else:  # 'NATIVE', for other model that do not have built-in chat_template
        """ example dialog after applying chat template:

        <s><|system|>\nYou are a helpful AI assistant for travel tips and recommendations</s>\n

        <s><|user|>\nWhat is France's capital?\n

        <|assistant|>\nBonjour! The capital of France is Paris!</s>\n

        <s><|user|>\nWhat can I do there?\n

        <|assistant|>\nParis, the City of Light...</s>\n

        <s><|user|>\nGive me a detailed list of the attractions I should visit.\n

        <|assistant|>\n</s>\n --> inference == True
        """
        # chat_template =  "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
        encode = partial(tokenizer.encode, add_special_tokens=False, truncation=dataset_config.truncation, max_length=dataset_config.max_length)
        system_prompt = []
        if dialog[0]['role'] == 'system':
            system_message = dialog[0]['content']
            dialog = dialog[1:]
            system_prompt = [tokenizer.bos_token_id] + encode(f"{S_INST}\n{system_message}") + [tokenizer.eos_token_id] + encode("\n")

        prompt_tokens = [(system_prompt if i == 0 else []) + [tokenizer.bos_token_id] + encode(f"{U_INST}\n{(prompt['content']).strip()}\n") for i, prompt in enumerate(dialog[::2])]
        answer_tokens = [encode(f"{A_INST}\n{answer['content'].strip()}") + [tokenizer.eos_token_id] for answer in dialog[1::2]]

        if inference:
            if len(prompt_tokens) != len(answer_tokens) + 1:
                raise ValueError("Prompt length should be one more than answer length for inference. e.g. User: Assistant: User: ")
            generation_prompt = encode(f"{A_INST}\n")
            answer_tokens.append(generation_prompt)

    # combine prompt and answer tokens
    dialog_tokens = list(itertools.chain.from_iterable(zip(prompt_tokens, answer_tokens)))

    # add labels, convert prompt token to -100 in order to ignore in loss function if pretrain is False, else just as is
    if dataset_config.pretrain:
        labels_tokens = [c for c in dialog_tokens]
    else:
        labels_tokens = [len(c)*[-100,] if i % 2 == 0 else c for i, c in enumerate(dialog_tokens)]

    # combine prompt and answer tokens and labels
    combined_tokens = {
        "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
        "labels": list(itertools.chain(*(t for t in labels_tokens))),
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))
