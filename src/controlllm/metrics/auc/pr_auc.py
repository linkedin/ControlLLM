import numpy as np
import datasets
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

# Try to import evaluate; if unavailable, create a dummy evaluate module for testing purposes.
try:
    import evaluate
except ImportError:
    class DummyMetric:
        pass

    class DummyFileUtils:
        @staticmethod
        def add_start_docstrings(desc, kwargs):
            def decorator(cls):
                return cls
            return decorator

    class DummyEvaluate:
        Metric = DummyMetric
        utils = type("utils", (), {"file_utils": DummyFileUtils})

    evaluate = DummyEvaluate()

_DESCRIPTION = """
This metric computes the area under the Precision-Recall Curve (PR AUC), also known as Average Precision.
The PR AUC summarizes the trade-off between precision and recall for different probability thresholds,
providing an aggregate measure of performance. This metric is particularly useful for imbalanced datasets,
where it emphasizes the performance on the positive class.
"""

_KWARGS_DESCRIPTION = """
Args:
- references (array-like of shape (n_samples,) or (n_samples, n_classes)): Ground truth labels.
    - binary: expects an array-like of shape (n_samples,)
    - multiclass: expects an array-like of shape (n_samples,)
    - multilabel: expects an array-like of shape (n_samples, n_classes)
- predictions (array-like of shape (n_samples,) or (n_samples, n_classes)): Model prediction scores.
    - binary: expects an array-like of shape (n_samples,)
    - multiclass: expects an array-like of shape (n_samples, n_classes) where each entry corresponds to a class score.
    - multilabel: expects an array-like of shape (n_samples, n_classes)
- average (str): Specifies the type of averaging performed on the data. Defaults to 'macro'. Options are:
    - 'micro': Calculate metrics globally by considering each element of the label indicator matrix as a label. Only works with multilabel.
    - 'macro': Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    - 'weighted': Calculate metrics for each label, and find their average weighted by the number of true instances for each label.
    - 'samples': Calculate metrics for each instance, and find their average. Only works with multilabel.
    - None: No averaging is performed, and scores for each class are returned. Only works with multilabel.
- sample_weight (array-like of shape (n_samples,)): Sample weights. Defaults to None.
Returns:
    pr_auc (float or array-like of shape (n_classes,)): If an average is provided, returns a float;
        otherwise, returns an array of scores for each class.
"""

_CITATION = """\
@article{scikit-learn,
title={Scikit-learn: Machine Learning in {P}ython},
author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
journal={Journal of Machine Learning Research},
volume={12},
pages={2825--2830},
year={2011}
}
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PRAUC(evaluate.Metric):
    def _info(self):
        # Define features similarly to the ROC-AUC implementation.
        features = (
            {
                "predictions": datasets.Sequence(datasets.Value("float")),
                "references": datasets.Value("int32"),
            }
            if self.config_name == "multiclass"
            else {
                "references": datasets.Sequence(datasets.Value("int32")),
                "predictions": datasets.Sequence(datasets.Value("float")),
            }
            if self.config_name == "multilabel"
            else {
                "references": datasets.Value("int32"),
                "predictions": datasets.Value("float"),
            }
        )
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(features),
            reference_urls=[
                "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"
            ],
        )

    def _compute(
        self,
        references,
        predictions,
        average="macro",
        sample_weight=None,
    ):
        # For multiclass, binarize the references using one-vs-rest approach.
        if self.config_name == "multiclass":
            # Get sorted list of unique classes.
            classes = sorted(list(set(references)))
            # Binarize the ground truth labels.
            binary_refs = label_binarize(references, classes=classes)
            pr_auc = average_precision_score(
                binary_refs,
                predictions,
                average=average,
                sample_weight=sample_weight,
            )
        else:
            pr_auc = average_precision_score(
                references,
                predictions,
                average=average,
                sample_weight=sample_weight,
            )
        return {"pr_auc": pr_auc}


if __name__ == "__main__":
    # --- Testing the PRAUC metric implementation ---

    # Helper to simulate an instance of PRAUC with a given config.
    class DummyPRAUC(PRAUC):
        def __init__(self, config_name):
            self.config_name = config_name

    # Test 1: Binary classification
    print("Binary classification test:")
    binary_refs = [1, 0, 1, 1, 0, 0]
    binary_pred = [0.5, 0.2, 0.99, 0.3, 0.1, 0.7]
    metric_binary = DummyPRAUC(config_name="binary")
    result_binary = metric_binary._compute(binary_refs, binary_pred)
    print("PR AUC (binary):", result_binary["pr_auc"])

    # Test 2: Multiclass classification
    print("\nMulticlass classification test:")
    multiclass_refs = [1, 0, 1, 2, 2, 0]
    multiclass_pred = [
        [0.3, 0.5, 0.2],
        [0.7, 0.2, 0.1],
        [0.005, 0.99, 0.005],
        [0.2, 0.3, 0.5],
        [0.1, 0.1, 0.8],
        [0.1, 0.7, 0.2],
    ]
    metric_multiclass = DummyPRAUC(config_name="multiclass")
    result_multiclass = metric_multiclass._compute(multiclass_refs, multiclass_pred, average="macro")
    print("PR AUC (multiclass, macro):", result_multiclass["pr_auc"])

    # Test 3: Multilabel classification
    print("\nMultilabel classification test:")
    multilabel_refs = [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
    ]
    multilabel_pred = [
        [0.8, 0.2, 0.6],
        [0.3, 0.7, 0.9],
        [0.9, 0.6, 0.1],
        [0.2, 0.1, 0.8],
    ]
    metric_multilabel = DummyPRAUC(config_name="multilabel")
    result_multilabel = metric_multilabel._compute(multilabel_refs, multilabel_pred, average=None)
    print("PR AUC (multilabel):", result_multilabel["pr_auc"])
