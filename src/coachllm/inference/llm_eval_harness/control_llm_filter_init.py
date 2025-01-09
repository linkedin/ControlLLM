import sys
from functools import partial
from typing import List
from lm_eval.api.filter import Filter, FilterEnsemble
from lm_eval.api.registry import get_filter, register_filter

# Custom Filter Implementation
@register_filter("custom")
class CustomFilter(Filter):
    """
    Custom filter that applies a custom, user-defined function to the model responses.
    """

    def __init__(self, **kwargs) -> None:
        self.filter_fn = kwargs.pop("filter_fn")
        super().__init__(**kwargs)

    def apply(self, resps, docs):
        return self.filter_fn(resps, docs)

# Function to Build Filter Ensemble
def build_filter_ensemble(
    filter_name: str, components: List[List[str]]
) -> FilterEnsemble:
    """
    Create a filtering pipeline.
    """
    filters = []
    for function, kwargs in components:
        if kwargs is None:
            kwargs = {}
        # create a filter given its name in the registry
        f = partial(get_filter(function), **kwargs)
        # add the filter as a pipeline step
        filters.append(f)

    return FilterEnsemble(name=filter_name, filters=filters)

# Dynamically Monkey-Patch the `lm_eval.filter` Package
def monkey_patch_lm_eval():
    from lm_eval import filters as filter_package

    # Add custom imports and functionality
    filter_package.custom = CustomFilter
    filter_package.build_filter_ensemble = build_filter_ensemble

    # Update the sys.modules cache to ensure patched module is used
    sys.modules["lm_eval.filter"] = filter_package

# Apply the monkey patch
monkey_patch_lm_eval()
