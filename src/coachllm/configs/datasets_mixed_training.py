import os
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class AbstractDataset:
    dataset: str  # dataset name to be defined in child class
    prompt_columns: Optional[Union[str, List[str]]] = None  # encode as input_ids
    response_column: Optional[str] = None  # encode as labels
    dataset_cache_dir: str = "/shared/public/data/controlllm/cache/{dataset}/dataset/"  # cache for final preprocessed dataset/features
    avro_cache_dir: str = "/shared/public/data/controlllm/cache/{dataset}/avro/"  # cache for avro generators
    dataset_map_cache_dir: str = "/shared/public/data/controlllm/cache/{dataset}/hf_map/" # cache for huggingface map function
    hf_hub_dataset_cache_dir: str = "/shared/public/data/controlllm/datasets/"  # cache for huggingface public datasets

    local_test: bool = False  # enable local testing with mock data
    print_text: bool = False  # print the text while encoding the dataset
    pretrain: bool = False  # if True, apply -100 in prompt if instruction fine tuning else compute the loss for all tokens
    # mixed training means to train with mixed data of Pretrain and SFT(different from normal SFT, pretrained on SFT data means no masking on input prompt) while eval on SFT data with labels masked for input prompt, following mixed training note from https://arxiv.org/pdf/2309.14316
    mixed_training: bool = False  # if True, apply mixed training for SFT, only used for dataset with pretrain False.
    run_validation: bool = False  # whether to run validation during training, suggest to disable it for pretraining dataset and enable it for SFT dataset
    include_val: bool = False  # include validation set in training, this makes sense in pretraining
    sample_size: Union[int, float] = 1.00  # sample size of the dataset used for training and evaluation(1.0 means full dataset, 0.1 means 10% of the dataset, 3000 means 3000 samples). Note that it also supports upsamping by setting it to a float number greater than 1.0 which int means absolute sample size (e.g. 2 means 2 samples, 2.0 means 2 times of the original dataset)
    test_split_ratio: float = -1  # split ratio for validation set. Only used when run_validation is False and train_split and test_split are the same. Recommendation - SFT: when train_split == test_split, set it to 0.2 or 0.1. Pretrain: set to -1 to use the full dataset for training and testing(don't worry, train_confg.max_eval_step will control the number of steps for evaluation)

    # tokenizing settings in data preprocessing
    # truncate the prompt + answer to max_length in tokenizing, safe to be False when batch_stategy is packing
    truncation: bool = False
    max_length: int = None  # max length of the tokenized sequence

    # chat template for SFT, for detailed format, see documentation in ./src/controlllm/data/utils.py -> tokenize_dialog()
    chat_template: str = "LLAMA3"  # "LLAMA3, "LLAMA2", "MISTRAL", or "NATIVE", note that native is our own format

    # dataset preprocessing
    # force refresh the cache if True, useful if the data in hdfs is updated
    force_refresh: bool = False
    # number of workers for processing the dataset, < multiprocessing.cpu_count()
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)
    # number of batches for packing strategy, default to 2 * num_workers_processing
    num_batches: Optional[int] = max(1, multiprocessing.cpu_count() // 4) * 4
    # drop last, useful for packing strategy, if True, drop the last batch if it's not full
    drop_last: bool = False

    def __post_init__(self):
        self.dataset_cache_dir = self.dataset_cache_dir.format(dataset=self.dataset)
        self.avro_cache_dir = self.avro_cache_dir.format(dataset=self.dataset)
        self.dataset_map_cache_dir = self.dataset_map_cache_dir.format(dataset=self.dataset)
        if not os.path.exists(self.dataset_cache_dir):
            os.makedirs(self.dataset_cache_dir)
        if not os.path.exists(self.avro_cache_dir):
            os.makedirs(self.avro_cache_dir)
        if not os.path.exists(self.dataset_map_cache_dir):
            os.makedirs(self.dataset_map_cache_dir)


@dataclass
class Ultrachat200k(AbstractDataset):
    dataset: str = "HuggingFaceH4/ultrachat_200k"
    prompt_columns: Optional[str] = "messages"  # for SFT
    response_column: Optional[str] = "messages"  # for SFT
    train_split: str = "train_sft"
    test_split: str = "test_sft"
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class UltraFeedback(AbstractDataset):
    dataset: str = "penbmb/UltraFeedback"
    train_split: str = "train"
    test_split: str = "validation"
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class samsum_dataset(AbstractDataset):
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    pretrain: bool = False
    trust_remote_code: bool = False


@dataclass
class OpenAssistantDataset(AbstractDataset):
    dataset: str =  "OpenAssistant/oasst1"
    train_split: str = "train"
    test_split: str = "validation"
    sample_size: Union[int, float] = 0.5  # OpenAssistant has 80k, sample it to 40k
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class OpenOrcaDataset(AbstractDataset):
    # note that the dataset name is different from Open-Orca/OpenOrca in the original code to Open-Orca/open_orca, to enable loading from cache
    dataset: str = "Open-Orca/open_orca"
    system_column: str = "system_prompt"  # for SFT
    prompt_columns: Optional[str] = "question"  # for SFT
    response_column: Optional[str] = "response"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    sample_size: Union[int, float] = 0.01  # OpenOrca has 4.2 million, sample it to 42k
    train_split: str = "train"
    test_split: str = "train"
    pretrain: bool = False
    run_validation: bool = False



@dataclass
class RedPajamaDataset(AbstractDataset):
    # togethercomputer/RedPajama-Data-1T -> togethercomputer/redpajama-data-1t to use cache
    dataset: str = "togethercomputer/redpajama-data-1t"
    prompt_columns: Optional[str] = "text"  # for SFT
    response_column: Optional[str] = "text"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    sample_size: Union[int, float] = 0.0001  # togethercomputer/RedPajama-Data-1T, sample it to 100K
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"
    pretrain: bool = True
    run_validation: bool = False


@dataclass
class grammar_dataset(AbstractDataset):
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class alpaca_dataset(AbstractDataset):
    dataset: str = "llamafactory/alpaca_en"
    system_column: str = "instruction"  # for SFT
    prompt_columns: Optional[str] = "input"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    sample_size: Union[int, float] = 1.0  # llamafactory/alpaca_en has 51k, include all
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class UltraInteract_sft(AbstractDataset):
    # note that the dataset name is different from the one in the original code, to enable loading from cache
    dataset: str = "openbmb/ultra_interact_sft"   
    prompt_columns: Optional[str] = "instruction"  # for SFT
    response_column: Optional[str] = "response"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    sample_size: Union[int, float] = 0.1  # UltraInteract_sft has 289k, sample it to 28k
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class MagpieDataset(AbstractDataset):
    # note that the dataset name is different from the one in the original code, to enable loading from cache, Magpie-Pro-MT-300K-v0.1->magpie-pro-mt-300_k-v0.1
    dataset: str = "Magpie-Align/magpie-pro-mt-300_k-v0.1"
    prompt_columns: Optional[str] = "conversations"  # for SFT
    response_column: Optional[str] = "conversations"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    sample_size: Union[int, float] = 0.1  # MagpieDaMagpie-Pro-MT-300Ktaset has 300K, sample it to 30K
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class OpenHermesDataset(AbstractDataset):
    # note that the dataset name is different from teknium/OpenHermes-2.5 in the original code to open_hermes-2.5, to enable loading from cache
    dataset: str = "teknium/open_hermes-2.5"
    prompt_columns: Optional[str] = "conversations"  # for SFT
    response_column: Optional[str] = "conversations"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    sample_size: Union[int, float] = 0.03  # OpenHermes has 1 million, sample it to 30K
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    run_validation: bool = False


@dataclass
class LlamaSynEZhDataset(AbstractDataset):
    def default_datafiles_sys_e():
        return ['book_cn', 'encyclopedia_cn', 'qa_forum_cn', 'web_cn', 'code_en', 'math_en', 'synthesis_en']

    # Llama-3-SynE-Dataset -> survivi/llama-3-syn_e-dataset to use cache
    dataset: str = "survivi/llama-3-syn_e-dataset"
    prompt_columns: Optional[str] = "text"  # for SFT
    response_column: Optional[str] = "text"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # survivi/Llama-3-SynE-Dataset has 168M
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: Optional[List[str]] = field(default_factory=default_datafiles_sys_e)
    test_split: Optional[List[str]] = field(default_factory=default_datafiles_sys_e)
    pretrain: bool = True
    run_validation: bool = False


@dataclass
class Llama3ChineseDataset(AbstractDataset):
    dataset: str = "Llama3ChineseDataset"
    prompt_columns: Optional[str] = "dialog"  # for SFT
    response_column: Optional[str] = "dialog"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # Llama3ChineseDataset has 1.6M
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    # dataset in chat template format with role and content
    train_split: str = "/shared/public/data/controlllm/datasets/sft_zh_with_all"
    test_split: str = "/shared/public/data/controlllm/datasets/sft_zh_with_all"  # there is no test split in this dataset
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.005  # 0.5% of the dataset for testing which is 1.6M * 0.005 = 8K


@dataclass
class AlpacaZhDataset(AbstractDataset):
    # note that the dataset name is different from the one in the original code, to enable loading from cache
    dataset: str = "llamafactory/alpaca_zh"
    system_column: str = "instruction"  # for SFT
    prompt_columns: Optional[str] = "input"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # llamafactory/alpaca_zh has 51k, include all
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.05 # 5% of the dataset for testing which is 51k * 0.05 = 2.55K


@dataclass
class AlpacaGPT4ZhDataset(AbstractDataset):
    dataset: str = "llamafactory/alpaca_gpt4_zh"
    system_column: str = "instruction"  # for SFT
    prompt_columns: Optional[str] = "input"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # llamafactory/alpaca_zh has 42k, include all
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.05  # 5% of the dataset for testing which is 42k * 0.05 = 2.1K


@dataclass
class RuozhibaZhDataset(AbstractDataset):
    dataset: str = "hfl/ruozhiba_gpt4"
    system_column: str = "instruction"  # for SFT
    prompt_columns: Optional[str] = "input"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # hfl/ruozhiba_gpt4 has 45k, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.05  # 5% of the dataset for testing which is 45k * 0.05 = 2.25K


@dataclass
class OaastSFTZhDataset(AbstractDataset):
    dataset: str = "OaastSFTZhDataset"
    system_column: str = "instruction"  # for SFT
    prompt_columns: Optional[str] = "input"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    history_column: Optional[str] = "history"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # OaastSFTZhDataset has 80k, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    # dataset in chat template format with role and content
    train_split: str = "/shared/public/data/controlllm/datasets/oaast_sft_zh"
    test_split: str = "/shared/public/data/controlllm/datasets/oaast_sft_zh"  # there is no test split in this dataset
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.05 # 5% of the dataset for testing which is 80k * 0.05 = 4K


@dataclass
class StemZhDataset(AbstractDataset):
    dataset: str = "hfl/stem_zh_instruction"
    system_column: str = "instruction"  # for SFT
    prompt_columns: Optional[str] = "input"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # hfl/stem_zh_instruction has 256k, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.01  # 1% of the dataset for testing which is 256k * 0.01 = 2.56K


@dataclass
class OpenMathInstruct2Dataset(AbstractDataset):
    # note that the dataset name is different from the one in the original code, to enable loading from cache, nvidia/OpenMathInstruct-2 -> nvidi/open_math_instruct-2
    dataset: str = "nvidia/open_math_instruct-2"
    prompt_columns: Optional[str] = "problem"  # for SFT
    response_column: Optional[str] = "generated_solution"  # for SFT
    problem_source_column: Optional[str] = "problem_source"  # for SFT
    expected_answer_column: Optional[str] = "expected_answer"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # nvidia/OpenMathInstruct-2 has 14m, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.001  # 0.1% of the dataset for testing which is 14m * 0.001 = 14K


@dataclass
class OpenCoderCPTAnnealing(AbstractDataset):
    def default_names_open_coder_annealing():
        return ["synthetic_code_snippet", "synthetic_qa"]

    dataset: str = "OpenCoder-LLM/opc-annealing-corpus"
    names: List[str] = field(default_factory=default_names_open_coder_annealing)
    prompt_columns: Optional[str] = "text"  # for pretraining
    response_column: Optional[str] = "text"  # for pretraining
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # OpenCoder-LLM/opc-annealing-corpus has ~2m, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = True
    mixed_training: bool = False
    run_validation: bool = True
    test_split_ratio: float = 0.001  # 0.1% of the dataset for testing which is 2m * 0.001 = 2K


@dataclass
class OpenCoderSFTStage1(AbstractDataset):
    def default_names_open_coder_stage1():
        return ["filtered_infinity_instruct", "largescale_diverse_instruct", "realuser_instruct"]

    dataset: str = "OpenCoder-LLM/opc-sft-stage1"
    names: List[str] = field(default_factory=default_names_open_coder_stage1)
    prompt_columns: Optional[str] = "instruction"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # OpenCoder-LLM/opc-sft-stage1 has ~4.2m, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.001  # 0.1% of the dataset for testing which is 4.2m * 0.001 = 4.2K


@dataclass
class OpenCoderSFTStage2(AbstractDataset):
    def default_names_open_coder_stage2():
        return ["educational_instruct", "evol_instruct", "mceval_instruct", "package_instruct"]

    dataset: str = "OpenCoder-LLM/opc-sft-stage2"
    names: List[str] = field(default_factory=default_names_open_coder_stage2)
    prompt_columns: Optional[str] = "instruction"  # for SFT
    response_column: Optional[str] = "output"  # for SFT
    include_val: bool = False  # there is no train/val split in this dataset yet
    force_refresh: bool = False
    sample_size: Union[int, float] = 1.0  # nvidia/OpenMathInstruct-2 has 445.9k, include all
    max_workers: int = max(1, multiprocessing.cpu_count() // 16)  # 1/16 of the cpu cores due to large side of the dataset
    train_split: str = "train"
    test_split: str = "train"  # there is no test split in this dataset yet
    pretrain: bool = False
    mixed_training: bool = True
    run_validation: bool = True
    test_split_ratio: float = 0.01  # 0.1% of the dataset for testing which is 445.9k * 0.01 = 4.459K
