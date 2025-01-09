# overwrite the CrossEntropyLoss with LigerCrossEntropyFunction
from liger_kernel.transformers import (
    apply_liger_kernel_to_llama,
    apply_liger_kernel_to_mistral
)

# apply all kernel customizations - with fused_linear_cross_entropy, we could double batch size for llama due to large vocab that consumes more GPU memory
def apply_model_customizations():
    apply_liger_kernel_to_llama(
        rope=True,
        rms_norm=True,
        cross_entropy=False,
        swiglu=True,
        fused_linear_cross_entropy=True,
    )
    apply_liger_kernel_to_mistral()
