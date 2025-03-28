import sys
import importlib.machinery
import importlib.abc
import importlib.util


class CustomEvaluatorsLoader(importlib.abc.Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def create_module(self, spec):
        if hasattr(self.original_loader, "create_module"):
            return self.original_loader.create_module(spec)
        return None

    def exec_module(self, module):
        # Execute the original module code.
        self.original_loader.exec_module(module)

        # Override RetrievalEvaluator with your custom class.
        if module.__name__ == "mteb.evaluation.evaluators":
            from controlllm.inference.llm_eval_mteb.RetrievalEvaluator import RetrievalEvaluator as CustomRetrievalEvaluator
            module.RetrievalEvaluator = CustomRetrievalEvaluator

        # Override AbsTaskRetrieval inside its module.
        elif module.__name__ == "mteb.abstasks.AbsTaskRetrieval":
            from controlllm.inference.llm_eval_mteb.AbsTaskRetrieval import AbsTaskRetrieval as CustomAbsTaskRetrieval
            module.AbsTaskRetrieval = CustomAbsTaskRetrieval

        # Ensure `mteb.abstasks` gets the updated `AbsTaskRetrieval`
        elif module.__name__ == "mteb.abstasks":
            from controlllm.inference.llm_eval_mteb.AbsTaskRetrieval import AbsTaskRetrieval as CustomAbsTaskRetrieval
            import mteb.abstasks.AbsTaskRetrieval as AbsTaskRetrievalModule
            module.AbsTaskRetrieval = CustomAbsTaskRetrieval
            AbsTaskRetrievalModule.AbsTaskRetrieval = CustomAbsTaskRetrieval


class CustomEvaluatorsFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in {"mteb.evaluation.evaluators", "mteb.abstasks.AbsTaskRetrieval", "mteb.abstasks"}:
            # Use the default PathFinder to get the spec.
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            # Wrap its loader with our custom loader.
            spec.loader = CustomEvaluatorsLoader(spec.loader)
            return spec
        return None


# Insert our custom finder at the beginning of the meta path.
sys.meta_path.insert(0, CustomEvaluatorsFinder())
