import os
import re
from pathlib import Path
from typing import List, Union
from huggingface_hub import snapshot_download

from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(model_name="meta-llama/Llama-3.1-8B-Instruct") -> PreTrainedTokenizer:
    # get current file path CURRENT_FILE_PATH, use its parent directory for model download
    CURRENT_FILE_PATH = Path(__file__).absolute()
    # If model_name is an absolute path(starts with / on Unix-like systems), CURRENT_FILE_PATH.parent.joinpath(model_name) will effectively return model_name as is, ignoring CURRENT_FILE_PATH.parent.
    models_path = CURRENT_FILE_PATH.parent.joinpath(model_name)
    models_path.mkdir(parents=True, exist_ok=True)

    # Check if it is already downloaded
    if not len(list(models_path.glob("*.json"))) > 0:
        # Download all files from the repository
        snapshot_download(repo_id=model_name, allow_patterns=["*.json"], local_dir=models_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(models_path))
    return tokenizer


def to_dialog(user_prompt=None, response=None):
    dialog = []
    if user_prompt:
        dialog.append({
                "role": "user",
                "content": user_prompt,
        })
    if response:
        dialog.append({
                "role": "assistant",
                "content": response,
        })
    return dialog


class PromptTemplateApplier:
    def __init__(self, model_name=os.environ['MODEL_PATH']):
        self.tokenizer: PreTrainedTokenizer = get_tokenizer(model_name)

    def apply_prompt_template(self, user_prompt: Union[List[str], str], response: Union[List[str], str]=None, add_generation_prompt=True, add_bos_token=True):
        if isinstance(user_prompt, list):
            dialogs = []
            for prompt, resp in zip(user_prompt, response):
                dialog = to_dialog(prompt, resp)
                dialogs.append(dialog)
            input_prompt = self.tokenizer.apply_chat_template(dialogs, add_generation_prompt=add_generation_prompt, tokenize=False)
            if add_bos_token is False:  # by default apply_chat_template will add bos_token such as <|begin_of_text|>
                input_prompt = input_prompt.replace(self.tokenizer.decode(self.tokenizer.bos_token_id), "")
        else:
            dialog = to_dialog(user_prompt, response)
            input_prompt = self.tokenizer.apply_chat_template(dialog, add_generation_prompt=add_generation_prompt, tokenize=False)
            if add_bos_token is False:
                input_prompt = input_prompt.replace(self.tokenizer.decode(self.tokenizer.bos_token_id), "")
            # llama added default system prompt which is not desired behavior here, remove it. refer to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/74
            # Define the default system prompt structure to match (excluding the specific date)
            default_system_prompt_to_be_removed_pattern = r"<\|start_header_id\|>system<\|end_header_id\|>\n\nCutting Knowledge Date: .*\nToday Date: \d{2} \w{3} \d{4}\n\n<\|eot_id\|>"
            # Remove the matching system prompt from input_prompt
            input_prompt = re.sub(default_system_prompt_to_be_removed_pattern, "", input_prompt)

        return input_prompt
