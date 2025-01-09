# Most of the code taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/cddce0a148ec1710e2d60546c6f92727dd8a78fd/lm_eval/tasks/leaderboard/math/utils.py
import re
import signal
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger
from controlllm.data.open_math_instruct_dataset import apply_prompt
from controlllm.inference.llm_eval_harness.additional_tasks.tokenizer import PromptTemplateApplier
from controlllm.inference.llm_eval_harness.additional_tasks.gsm8k.few_shot_examples import examples_map as examples_gsm8k

try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )

# taken from
# https://github.com/wellecks/lm-evaluation-harness/blob/master/lm_eval/tasks/minerva_math.py
llama_eval_instruction = 'Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.'
meta_math_instruction = 'Your response should end with \"The final answer is \\boxed{{[answer]}}\" where [answer] is the response to the problem.'

# note that the fine tuned version based on dataset of meta_math_instruction has \\boxed{{[answer]}}, but original baseline model is trained with [answer]
def doc_to_text(doc: dict) -> str:
    input_final_prompts = doc["input_final_prompts"][0]
    if llama_eval_instruction in input_final_prompts:
        input_final_prompts.replace(llama_eval_instruction, meta_math_instruction)
    return input_final_prompts


# note that the fine tuned version based on dataset of meta_math_instruction has \\boxed{{[answer]}}, but original baseline model is trained with [answer]
def doc_to_text_baseline(doc: dict) -> str:
    input_final_prompts = doc["input_final_prompts"][0]
    return input_final_prompts


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        # note that the orginal gsm8k data does not follow boxed format in the solution, so use the extracted answer in input_correct_responses directly as group truth here
        # compared to additional_tasks->math->utils.py, we used the boxed string lookup from solution as groundtruth
        out_doc = {
            "problem": doc["input_question"],
            "solution": doc["solution"] if "solution" in doc else doc["input_correct_responses"][0],
            "answer": doc["input_correct_responses"][0],  # note that in meta eval dataset, input_correct_responses is: [ "11", "11.", "11.0", "11.0.", "11.00", "11.00.", "11", "11.", "11.0", "11.0.", "11.00", "11.00." ], we take the first one for exact match. TODO: may need to normalize this to account for model's variations
            "meta_target": doc["input_correct_responses"]
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    processed_doc = dataset.map(_process_doc)
    # # take the first 10 to debug
    # processed_doc = processed_doc.select(range(10))
    return processed_doc


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    # this is the implementation of "is output correct" from https://github.com/Kipok/NeMo-Skills
    # answer = extract_answer(candidates)

    # if is_output_correct(answer, doc["answer"]):
    #     retval = 1
    # else:
    #     retval = 0

    # this is the implementation of "is output correct" from lm-evluation-harness(same from https://github.com/meta-llama/llama-recipes/blob/main/tools/benchmarks/llm_eval_harness/meta_eval/meta_template/math_hard/utils.py)
    last_boxed_string = last_boxed_only_string(candidates)
    if not last_boxed_string:
        # No boxed string found, so we can't evaluate
        return {"exact_match": 0}
    unnormalized_answer = remove_boxed(last_boxed_string)
    answer = normalize_final_answer(unnormalized_answer)

    if answer.strip() == (doc["answer"].strip() if isinstance(doc["answer"], str) else str(doc["answer"])) or is_equiv(answer, str(doc["answer"])):
        retval = 1
    else:
        retval = 0

    results = {
        "exact_match": retval,
    }
    return results


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                eval_logger.debug(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        eval_logger.debug(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        eval_logger.error(e)
        raise
    except Exception as e:
        eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    else:
        return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """
    Normalize a final answer to a quantitative reasoning question.

    Copied character for character from appendix D of Lewkowycz et al. (2022)
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract answer that is in LaTeX math, is bold,
    # is surrounded by a box, etc.
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize 100,000 -> 100000
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


# 8 examples are from lm-evluation-harness's tasks/gsm8k/gsm8k-cot-llama.yaml. Note that answer in the solution is not boxed.
def list_fewshot_samples_baseline(apply_chat_template=True, model_name="meta-llama/Llama-3.1-8B-Instruct") -> list[dict]:
    fewshot_samples = [
        {
            "problem": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "solution": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The final answer is 6",
            "answer": "6",
            "few_shot": "1",
        },
        {
            "problem": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "solution": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The final answer is 5",
            "answer": "5",
            "few_shot": "1",
        },
        {
            "problem": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "solution": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The final answer is 39",
            "answer": "39",
            "few_shot": "1",
        },
        {
            "problem": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
            "solution": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The final answer is 8",
            "answer": "8",
            "few_shot": "1",
        },
        {
            "problem": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
            "solution": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The final answer is 9",
            "answer": "9",
            "few_shot": "1",
        },
        {
            "problem": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
            "solution": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29",
            "answer": "29",
            "few_shot": "1",
        },
        {
            "problem": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
            "solution": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33",
            "answer": "33",
            "few_shot": "1",
        },
        {
            "problem": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
            "solution": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8",
            "answer": "8",
            "few_shot": "1",
        },
    ]

    if apply_chat_template and model_name is not None:
        prompt_template_applier = PromptTemplateApplier(model_name=model_name)
        for fewshot_sample in fewshot_samples:
            fewshot_sample["problem"] = apply_prompt(fewshot_sample["problem"], "math")
            fewshot_sample["input_final_prompts"] = [prompt_template_applier.apply_prompt_template(fewshot_sample["problem"], None, add_generation_prompt=False, add_bos_token=False)]
            fewshot_sample["solution"] = prompt_template_applier.apply_prompt_template(None, fewshot_sample["solution"], add_generation_prompt=False, add_bos_token=False)

    return fewshot_samples


# 8 examples are from NEMO-SKILLS/nemo_skills/prompt/few_shot_examples/gsm8k.py. Note that answer in the solution is boxed since our newly trained model with meta_math_instruction has \\boxed{{[answer]}}.
def list_fewshot_samples(apply_chat_template=True, model_name="meta-llama/Llama-3.1-8B-Instruct") -> list[dict]:
    fewshot_samples = examples_gsm8k["gsm8k_standard_few_shot"]

    if apply_chat_template and model_name is not None:
        prompt_template_applier = PromptTemplateApplier(model_name=model_name)
        for fewshot_sample in fewshot_samples:
            fewshot_sample["problem"] = apply_prompt(fewshot_sample["problem"], "math")
            fewshot_sample["input_final_prompts"] = [prompt_template_applier.apply_prompt_template(fewshot_sample["problem"], None, add_generation_prompt=False, add_bos_token=False)]
            fewshot_sample["solution"] = prompt_template_applier.apply_prompt_template(None, fewshot_sample["solution"], add_generation_prompt=False, add_bos_token=False)

    return fewshot_samples
