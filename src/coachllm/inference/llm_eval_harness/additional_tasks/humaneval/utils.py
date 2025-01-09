# code enhanced from https://github.com/EleutherAI/lm-evaluation-harness/pull/1992/files
import datasets
from lm_eval import utils

from controlllm.inference.llm_eval_harness.additional_tasks.tokenizer import PromptTemplateApplier
from controlllm.inference.llm_eval_harness.additional_tasks.hf_evaluate import initialize_metrics_modules


# # uncomment this to run simple test to check code execution is enabled before model generation
# test_cases = ["assert add(2, 3)==5"]
# candidates = [["def add(a,b): return a*b"]]
# results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1], num_workers=1)


def pass_at_1(references, predictions):
    pass_at_k = initialize_metrics_modules(module="code_eval")  # huggingface code_eval module is not thread safe, so we need to initialize it in each function call
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
        num_workers=16,
    )[0]["pass@1"]


def build_references(doc):
    return doc["test"] + "\n" + f"check({doc['entry_point']})"


def build_predictions(resps, docs):
    preds = []

    patterns = [
        "```",
        "\nclass",
        "\ndef",
        "\n#",
        "\nif",
        "\nprint"
    ]

    def sanitize_response(resp):
        if not isinstance(resp, str):
            return resp

        resp = resp.rstrip()  # remove the \n from the end of the response

        for pattern in patterns:
            if pattern in resp:  # we added ```python\n{{prompt}}, so the generated code is ending with ```
                resp = resp.split(pattern)[0]  # left split
            else:
                pass

        resp = resp.rstrip()  # remove the \n from the end of the response

        return resp

    for resp, doc in zip(resps, docs):
        pred = [doc["prompt"] + sanitize_response(r) for r in resp]
        preds.append(pred)

    return preds


def doc_to_text(doc: dict) -> str:
    return doc["input_final_prompt"]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = doc.copy()
        out_doc["input_final_prompt"] = apply_prompt(doc)
        return out_doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


def process_docs_w_chat_template(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = doc.copy()
        out_doc["input_final_prompt"] = apply_chat_template(doc)
        return out_doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


# without apply chat_template
def apply_prompt(doc: dict) -> str:
    # prompt template followed: https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__human_eval_plus__details?row=0
    # e.g. "<|start_header_id|>user<|end_header_id|>\n\nWrite a solution to the following problem and make sure that it passes the tests:\n```python\nfrom typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n separate those group into separate strings and return the list of those.\n Separate groups are balanced (each open brace is properly closed) and not nested within each other\n Ignore any spaces in the input string.\n >>> separate_paren_groups('( ) (( )) (( )( ))')\n ['()', '(())', '(()())']\n \"\"\"\n\n```<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere is the completed function:\n```python\nfrom typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n \"\"\" Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n separate those group into separate strings and return the list of those.\n Separate groups are balanced (each open brace is properly closed) and not nested within each other\n Ignore any spaces in the input string.\n >>> separate_paren_groups('( ) (( )) (( )( ))')\n ['()', '(())', '(()())']\n \"\"\""
    prompt_template = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{{prompt}}"  # as is: no prompt template
    # TODO: apply prompts from evalplus
    # Model instructions
    # instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    # response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    # if evalperf_type == "perf-instruct":
    #     instruction_prefix = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
    #     response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    # elif evalperf_type == "perf-CoT":
    #     instruction_prefix = "Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:"
    #     response_prefix = "Below is a Python script with a self-contained function that efficiently solves the problem and passes corresponding tests:"
    # elif evalperf_type is not None and evalperf_type != "instruct":
    #     raise ValueError(f"Invalid evalperf_type: {evalperf_type}")

    prompt = utils.apply_template(prompt_template, doc)

    return prompt


# with apply chat_template
def apply_chat_template(doc: dict) -> str:
    prompt = apply_prompt(doc)

    prompt += "\n\n```"  # follow prompt template at: https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__human_eval_plus__details?row=0

    prompt_template_applier = PromptTemplateApplier()
    input_final_prompt = prompt_template_applier.apply_prompt_template(prompt, None, add_generation_prompt=True, add_bos_token=False)

    # prompt template followed: https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__human_eval_plus__details?row=3
    user_prompt_template = "Here is the completed function:\n```python\n{{prompt}}"
    user_prompt = utils.apply_template(user_prompt_template, doc)

    input_final_prompt += user_prompt  # add prefix for assistant's response

    return input_final_prompt
