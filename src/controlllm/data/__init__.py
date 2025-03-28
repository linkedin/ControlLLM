from functools import partial

from controlllm.data.samsum_dataset import get_preprocessed_samsum as get_samsum_dataset
from controlllm.data.open_math_instruct_dataset import get_dataset as get_open_math_instruct_dataset
from controlllm.data.open_coder_cpt_dataset import get_dataset as get_open_coder_cpt_dataset
from controlllm.data.open_coder_instruct_dataset import get_dataset as get_open_coder_dataset
from controlllm.data.openassistant_dataset import get_dataset as get_openassistant_dataset
from controlllm.data.open_orca_dataset import get_dataset as get_open_orca_dataset
from controlllm.data.alpaca_dataset import get_dataset as get_alpaca_dataset
from controlllm.data.ultrachat_dataset import get_dataset as get_ultrachat_dataset
from controlllm.data.ultrainteract_dataset import get_dataset as get_ultrainteract_dataset
from controlllm.data.magpie_dataset import get_dataset as get_magpie_dataset
from controlllm.data.openhermes_dataset import get_dataset as get_openhermes_dataset
from controlllm.data.synezh_dataset import get_dataset as get_synezh_dataset
from controlllm.data.chatcompletion_dataset import get_dataset as get_chatcompletion_dataset
from controlllm.data.semantic_search_dataset import get_dataset as get_semantic_search_dataset, get_data_collator as get_semantic_search_data_collator, postprocess_dataset as postprocess_semantic_search_dataset
from controlllm.data.semantic_search_cosent_dataset import get_dataset as get_semantic_search_cosent_dataset, postprocess_dataset as postprocess_semantic_search_cosent_dataset


# map the dataset to its preprocessor
DATASET_PREPROC = {
    "llamafactory/alpaca_en": partial(get_alpaca_dataset),
    "llamafactory/alpaca_zh": partial(get_alpaca_dataset),
    "llamafactory/alpaca_gpt4_zh": partial(get_alpaca_dataset),
    "hfl/ruozhiba_gpt4": partial(get_alpaca_dataset),
    "hfl/stem_zh_instruction": partial(get_alpaca_dataset),
    "OaastSFTZhDataset": partial(get_alpaca_dataset),
    "Llama3ChineseDataset": get_chatcompletion_dataset,
    "survivi/llama-3-syn_e-dataset": get_synezh_dataset,
    "nvidia/open_math_instruct-2": get_open_math_instruct_dataset,
    "OpenCoder-LLM/opc-annealing-corpus": get_open_coder_cpt_dataset,
    "OpenCoder-LLM/opc-sft-stage1": get_open_coder_dataset,
    "OpenCoder-LLM/opc-sft-stage2": get_open_coder_dataset,

    "OpenAssistant/oasst1": partial(get_openassistant_dataset),
    "Open-Orca/open_orca": get_open_orca_dataset,
    "samsum_dataset": get_samsum_dataset,
    "HuggingFaceH4/ultrachat_200k": get_ultrachat_dataset,
    "openbmb/ultra_interact_sft": get_ultrainteract_dataset,
    "Magpie-Align/magpie-pro-mt-300_k-v0.1": get_magpie_dataset,
    "teknium/open_hermes-2.5": get_openhermes_dataset,

    "MSMarcoDataset": partial(get_semantic_search_dataset),
    "MSMarcoCosentDataset": partial(get_semantic_search_cosent_dataset),
    # add more dataset mapping here
}


# map the dataset to its postrocessor
DATASET_POSTPROC = {
    "MSMarcoDataset": partial(postprocess_semantic_search_dataset),
    "MSMarcoCosentDataset": partial(postprocess_semantic_search_cosent_dataset),

    # add more dataset postprocess mapping here
}

DATALOADER_COLLATE_FUNC = {
    "MSMarcoDataset": get_semantic_search_data_collator,
    "MSMarcoCosentDataset": get_semantic_search_data_collator,

    # add more dataloader collate mapping here
}


__all__ = [
    "DATASET_PREPROC",
    "DATALOADER_COLLATE_FUNC"
    "sample_dataset"
]
