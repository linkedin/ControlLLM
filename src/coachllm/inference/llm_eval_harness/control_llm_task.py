import logging
import numpy as np
from typing import List

from lm_eval import utils
from lm_eval.api.task import ConfigurableTask


eval_logger = logging.getLogger("lm-eval")


class ControlLLMConfigTask(ConfigurableTask):
    """
    Customermize the process_results method to handle the requirement of coding task.
    """

    def process_results(self, doc, results):
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)

        result_dict = {}
        use_metric = list(self._metric_fn_list.keys())
        if self.OUTPUT_TYPE == "loglikelihood":
            results = results[0]
            ll, is_greedy = results
            return {
                **({"perplexity": ll} if "perplexity" in use_metric else {}),
                **({"acc": int(is_greedy)} if "acc" in use_metric else {}),
            }
        elif self.OUTPUT_TYPE == "loglikelihood_rolling":
            (loglikelihood,) = results
            _words = self.count_words(self.doc_to_target(doc))
            _bytes = self.count_bytes(self.doc_to_target(doc))
            return {
                **(
                    {"word_perplexity": (loglikelihood, _words)}
                    if "word_perplexity" in use_metric
                    else {}
                ),
                **(
                    {"byte_perplexity": (loglikelihood, _bytes)}
                    if "byte_perplexity" in use_metric
                    else {}
                ),
                **(
                    {"bits_per_byte": (loglikelihood, _bytes)}
                    if "bits_per_byte" in use_metric
                    else {}
                ),
            }
        elif self.OUTPUT_TYPE == "multiple_choice":
            lls, is_greedy = zip(*results)

            # retrieve choices in List[str] form, to compute choice lengths, etc.
            choices = self.doc_to_choice(doc)
            completion_len = np.array([float(len(i)) for i in choices])

            if (
                2 * len(choices) == len(lls)
                and "acc_mutual_info" in self._metric_fn_list.keys()
            ):
                # then we are doing mutual info.
                # this stores the "dryrun" / unconditional answer loglikelihoods
                lls_unconditional = lls[1::2]
                if len(lls_unconditional) != len(choices):
                    raise ValueError
                # and this stores our "regular" conditional loglikelihoods
                lls = lls[::2]

            pred = np.argmax(lls)
            pred_norm = np.argmax(lls / completion_len)

            if self.multiple_input:
                gold = self.doc_to_text(doc)
            else:
                gold = self.doc_to_target(doc)

            gold_index_error = False
            if isinstance(gold, list):
                gold = [i if i < len(choices) else -100 for i in gold]
                if -100 in gold:
                    gold_index_error = True
            else:
                if isinstance(gold, int):
                    gold = gold if gold < len(choices) else -100
                elif isinstance(gold, str):
                    gold = choices.index(gold) if gold in choices else -100

                if gold == -100:
                    gold_index_error = True

            if gold_index_error:
                eval_logger.warning(
                    f"Label index was not in within range of available choices,"
                    f"Sample:\n\n{doc}\n\n"
                )

            if self.multiple_target:
                acc = 1.0 if pred in gold else 0.0
                acc_norm = 1.0 if pred_norm in gold else 0.0
                exact_match = int(any([is_greedy[i] if i != -100 else 0 for i in gold]))
            else:
                acc = 1.0 if pred == gold else 0.0
                acc_norm = 1.0 if pred_norm == gold else 0.0
                # TODO: this gets score of 0 on arc_challenge for pythia-70m. need to test that this works properly
                exact_match = int(is_greedy[gold]) if gold != -100 else 0

            prob_norm = utils.softmax(lls)

            # TODO use keyword arguments to the metric?
            # gold, pred, norm stuff, the original lls,
            result_dict = {
                **({"acc": acc} if "acc" in use_metric else {}),
                **({"f1": (gold, pred)} if "f1" in use_metric else {}),
                **({"mcc": (gold, pred)} if "mcc" in use_metric else {}),
                **({"acc_norm": acc_norm} if "acc_norm" in use_metric else {}),
                **({"exact_match": exact_match} if "exact_match" in use_metric else {}),
                **(
                    {"brier_score": (gold, prob_norm)}
                    if "brier_score" in use_metric
                    else {}
                ),
            }

            if "acc_mutual_info" in use_metric:
                lls_mutual_info = [
                    ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional)
                ]
                acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
                result_dict["acc_mutual_info"] = acc_mutual_info

        elif self.OUTPUT_TYPE == "generate_until":
            gold = self.doc_to_target(doc)
            result = results[0]
            if self.config.doc_to_choice is not None:
                # If you set doc_to_choice,
                # it assumes that doc_to_target returns a number.
                choices = self.doc_to_choice(doc)
                gold = choices[gold]
            # we expect multiple_targets to be a list.
            elif self.multiple_target:
                gold = list(gold)
            # Control LLM customization
            # follow https://github.com/EleutherAI/lm-evaluation-harness/pull/1992/files for humaneval to work
            elif type(gold) is not type(result) and not isinstance(result, List):
                # cast gold to the same type as result
                gold = type(result)(gold)

            for metric in self._metric_fn_list.keys():
                if self.multiple_target:
                    # in the case where we have multiple targets,
                    # return true if any are true
                    # TODO: this may break for multipLe_target, non zero-or-1 metrics
                    scores = []
                    if not isinstance(gold, list):
                        # sometimes, a multiple_target dataset has exceptions where one doc has only one string answer
                        # print(gold)
                        gold = [gold]
                    if metric == "exact_match":
                        result = [result for _ in range(len(gold))]
                        scores = self._metric_fn_list[metric](
                            references=gold,
                            predictions=result,
                            **self._metric_fn_kwargs[metric],
                        )[metric]
                        result_score = 1.0 if scores > 0.0 else 0.0
                    else:
                        for gold_option in gold:
                            try:
                                result_score = self._metric_fn_list[metric](
                                    references=[gold_option],
                                    predictions=[result],
                                    **self._metric_fn_kwargs[metric],
                                )
                            except (
                                TypeError
                            ):  # TODO: this is hacky and I don't want to do it
                                result_score = self._metric_fn_list[metric](
                                    [gold_option, result]
                                )
                            if isinstance(result_score, dict):
                                # TODO: this handles the case where HF evaluate returns a dict.
                                result_score = result_score[metric]
                            scores.append(result_score)
                        if any(scores):
                            result_score = 1.0
                        else:
                            result_score = 0.0
                else:
                    try:
                        result_score = self._metric_fn_list[metric](
                            references=[gold],
                            predictions=[result],
                            **self._metric_fn_kwargs[metric],
                        )
                    except TypeError:  # needed for now in order to use a different interface between our own metrics and HF Evaluate metrics
                        result_score = self._metric_fn_list[metric]([gold, result])
                    if isinstance(result_score, dict):
                        # TODO: this handles the case where HF evaluate returns a dict.
                        result_score = result_score[metric]
                result_dict[metric] = result_score
        else:
            raise ValueError(
                f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of ",
                "'loglikelihood', 'loglikelihood_rolling', 'generate_until' or 'multiple_choice'",
            )

        return result_dict
