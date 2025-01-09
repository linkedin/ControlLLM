# code enhanced from https://github.com/EleutherAI/lm-evaluation-harness/pull/2247/files#diff-194dbdab3e17ca2994eb45bf69089791052210851cfd5cc94d55979712ef22eb
import os
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


def doc_to_text(doc: dict) -> str:
    return doc["input_final_prompt"]


def build_predictions(resps, docs):
    preds = []

    def sanitize_response(resp):
        if not isinstance(resp, str):
            return resp

        resp = resp.strip()  # remove the \n from the beginning and end of the response
        if "```" in resp:  # we added ```python\n{{prompt}}, so the generated code is ending with ```
            resp = resp.rsplit("```")[0]
        else:
            pass
        resp = resp.strip()  # remove the \n from the beginning and end of the response

        return resp

    for resp, doc in zip(resps, docs):
        pred = [sanitize_response(r) for r in resp]
        preds.append(pred)

    return preds


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        # mbpp full have column named text instead of prompt, while mbpp sanitized have prompt column, so make it consistent with prompt column
        if "prompt" not in doc and "text" in doc:
            doc["prompt"] = doc["text"]
        doc["input_final_prompt"] = apply_prompt(doc)
        if getattr(doc, "is_fewshot", None) is not None:
            doc["is_fewshot"] = True
        return doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


def process_docs_w_chat_template(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        # mbpp full have column named text instead of prompt, while mbpp sanitized have prompt column, so make it consistent with prompt column
        if "prompt" not in doc and "text" in doc:
            doc["prompt"] = doc["text"]
        doc["input_final_prompt"] = apply_chat_template(doc)
        if getattr(doc, "is_fewshot", None) is not None:
            doc["is_fewshot"] = True
        return doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


# without apply chat_template
def apply_prompt(doc: dict, with_suffix=True) -> str:
    # follows: https://huggingface.co/datasets/meta-llama/Llama-3.1-8B-Instruct-evals/viewer/Llama-3.1-8B-Instruct-evals__mbpp__details?row=0
    prompt_template = "You are an expert Python programmer, and here is your task: {{prompt}} Your code should pass following tests:\n\n{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}\n" + ("```python\n" if with_suffix else "")
    prompt = utils.apply_template(prompt_template, doc)

    return prompt


# with apply chat_template
def apply_chat_template(doc: dict) -> str:
    prompt = apply_prompt(doc, with_suffix=False)

    prompt_template_applier = PromptTemplateApplier()
    input_final_prompt = prompt_template_applier.apply_prompt_template(prompt, None, add_generation_prompt=True, add_bos_token=False)

    input_final_prompt += "```python\n"  # add prefix for assistant's response

    return input_final_prompt


def raw_fewshot_samples():
    return [
        {
            "task_id": 2,
            "prompt": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "prompt": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "prompt": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]


# 3 examples are from https://github.com/EleutherAI/lm-evaluation-harness/pull/2247/files#diff-194dbdab3e17ca2994eb45bf69089791052210851cfd5cc94d55979712ef22eb
def list_fewshot_samples(apply_chat_template=False, model_name=os.environ['MODEL_PATH']) -> list[dict]:
    fewshot_samples = raw_fewshot_samples()

    if apply_chat_template and model_name is not None:
        prompt_template_applier = PromptTemplateApplier(model_name=model_name)

    for fewshot_sample in fewshot_samples:
        if apply_chat_template and model_name is not None:
            prompt = apply_prompt(fewshot_sample, with_suffix=False)
            fewshot_sample["input_final_prompt"] = prompt_template_applier.apply_prompt_template(prompt, None, add_generation_prompt=False, add_bos_token=False)
            assistant_response = "```python\n" + fewshot_sample["code"] + "\n\n```"  # add prefix for assistant's response
            fewshot_sample["code"] = prompt_template_applier.apply_prompt_template(None, assistant_response, add_generation_prompt=False, add_bos_token=False)
        else:
            prompt = apply_prompt(fewshot_sample, with_suffix=True)
            fewshot_sample["input_final_prompt"] = prompt
            fewshot_sample["code"] += "\n\n```"
            # fewshot_sample["code"] as is, in yaml config doc_to_target, \n''' is appended to the code

    return fewshot_samples


# 3 examples are from https://github.com/EleutherAI/lm-evaluation-harness/pull/2247/files#diff-194dbdab3e17ca2994eb45bf69089791052210851cfd5cc94d55979712ef22eb
def list_fewshot_samples_w_chat_template(apply_chat_template=True, model_name=os.environ['MODEL_PATH']) -> list[dict]:
    return list_fewshot_samples(apply_chat_template, model_name)
