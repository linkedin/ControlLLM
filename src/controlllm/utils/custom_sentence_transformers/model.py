from __future__ import annotations
import weakref
from typing import Dict, List, Optional
from dataclasses import dataclass
from torch import FloatTensor, Tensor, nn
from collections import OrderedDict

from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from sentence_transformers import SentenceTransformer

from controlllm.utils.custom_sentence_transformers.pairwise_margin_loss import MarginPairwiseLoss
from controlllm.utils.custom_sentence_transformers.cosent_margin_loss import MarginCoSENTLoss


@dataclass
class SentenceTransformerOutput(ModelOutput):
    """
    Base class for sentence transformers model outputs.

    Args:
        loss (`FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        simiarities (`List[FloatTensor]` of shape `(batch_size,)`, *optional*, returned when `labels` is provided). e.g.:
            similarity_prompt_response (`FloatTensor` of shape `(batch_size,)`):
                The cosine similarity between the prompt(query) and the response(document) text.
            similarity_prompt_chosen (`FloatTensor` of shape `(batch_size,)`):
                The cosine similarity between the prompt and the chosen text.
            similarity_prompt_rejected (`FloatTensor` of shape `(batch_size,)`):
                The cosine similarity between the prompt and the rejected text.
    """

    loss: Optional[FloatTensor] = None
    similarities: Optional[Dict[str, FloatTensor]] = None


class CustomSentenceTransformer(SentenceTransformer):
    """
    Loads or creates a SentenceTransformer model that can be used to map sentences / text to embeddings.

    Args:
        model_name_or_path (str, optional): If it is a filepath on disc, it loads the model from that path. If it is not a path,
            it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model
            from the Hugging Face Hub with that name.
        modules (Iterable[nn.Module], optional): A list of torch Modules that should be called sequentially, can be used to create custom
            SentenceTransformer models from scratch.
        device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU
            can be used.
        prompts (Dict[str, str], optional): A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text.
            The prompt text will be prepended before any text to encode. For example:
            `{"query": "query: ", "passage": "passage: "}` or `{"clustering": "Identify the main category based on the
            titles in "}`.
        default_prompt_name (str, optional): The name of the prompt that should be used by default. If not set,
            no prompt will be applied.
        similarity_fn_name (str or SimilarityFunction, optional): The name of the similarity function to use. Valid options are "cosine", "dot",
            "euclidean", and "manhattan". If not set, it is automatically set to "cosine" if `similarity` or
            `similarity_pairwise` are called while `model.similarity_fn_name` is still `None`.
        cache_folder (str, optional): Path to store models. Can also be set by the SENTENCE_TRANSFORMERS_HOME environment variable.
        trust_remote_code (bool, optional): Whether or not to allow for custom models defined on the Hub in their own modeling files.
            This option should only be set to True for repositories you trust and in which you have read the code, as it
            will execute code present on the Hub on your local machine.
        revision (str, optional): The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        local_files_only (bool, optional): Whether or not to only look at local files (i.e., do not try to download the model).
        token (bool or str, optional): Hugging Face authentication token to download private models.
        use_auth_token (bool or str, optional): Deprecated argument. Please use `token` instead.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to. `None` does no truncation. Truncation is
            only applicable during inference when :meth:`SentenceTransformer.encode` is called.
        model_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers model.
            Particularly useful options are:

            - ``torch_dtype``: Override the default `torch.dtype` and load the model under a specific `dtype`.
              The different options are:

                    1. ``torch.float16``, ``torch.bfloat16`` or ``torch.float``: load in a specified
                    ``dtype``, ignoring the model's ``config.torch_dtype`` if one exists. If not specified - the model will
                    get loaded in ``torch.float`` (fp32).

                    2. ``"auto"`` - A ``torch_dtype`` entry in the ``config.json`` file of the model will be
                    attempted to be used. If this entry isn't found then next check the ``dtype`` of the first weight in
                    the checkpoint that's of a floating point type and use that as ``dtype``. This will load the model
                    using the ``dtype`` it was saved in at the end of the training. It can't be used as an indicator of how
                    the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.
            - ``attn_implementation``: The attention implementation to use in the model (if relevant). Can be any of
              `"eager"` (manual implementation of the attention), `"sdpa"` (using `F.scaled_dot_product_attention
              <https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html>`_),
              or `"flash_attention_2"` (using `Dao-AILab/flash-attention <https://github.com/Dao-AILab/flash-attention>`_).
              By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"`
              implementation.
            - ``provider``: If backend is "onnx", this is the provider to use for inference, for example "CPUExecutionProvider",
              "CUDAExecutionProvider", etc. See https://onnxruntime.ai/docs/execution-providers/ for all ONNX execution providers.
            - ``file_name``: If backend is "onnx" or "openvino", this is the file name to load, useful for loading optimized
              or quantized ONNX or OpenVINO models.
            - ``export``: If backend is "onnx" or "openvino", then this is a boolean flag specifying whether this model should
              be exported to the backend. If not specified, the model will be exported only if the model repository or directory
              does not already contain an exported model.

            See the `PreTrainedModel.from_pretrained
            <https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained>`_
            documentation for more details.
        tokenizer_kwargs (Dict[str, Any], optional): Additional tokenizer configuration parameters to be passed to the Hugging Face Transformers tokenizer.
            See the `AutoTokenizer.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained>`_
            documentation for more details.
        config_kwargs (Dict[str, Any], optional): Additional model configuration parameters to be passed to the Hugging Face Transformers config.
            See the `AutoConfig.from_pretrained
            <https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoConfig.from_pretrained>`_
            documentation for more details.
        model_card_data (:class:`~sentence_transformers.model_card.SentenceTransformerModelCardData`, optional): A model
            card data object that contains information about the model. This is used to generate a model card when saving
            the model. If not set, a default model card data object is created.
        backend (str): The backend to use for inference. Can be one of "torch" (default), "onnx", or "openvino".
            See https://sbert.net/docs/sentence_transformer/usage/efficiency.html for benchmarking information
            on the different backends.

    Example:
        ::

            from sentence_transformers import SentenceTransformer

            # Load a pre-trained SentenceTransformer model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Encode some texts
            sentences = [
                "The weather is lovely today.",
                "It's so sunny outside!",
                "He drove to the stadium.",
            ]
            embeddings = model.encode(sentences)
            print(embeddings.shape)
            # (3, 768)

            # Get the similarity scores between all sentences
            similarities = model.similarity(embeddings, embeddings)
            print(similarities)
            # tensor([[1.0000, 0.6817, 0.0492],
            #         [0.6817, 1.0000, 0.0421],
            #         [0.0492, 0.0421, 1.0000]])

            # For training, you can use the model with a custom loss function
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # workaround by making sentence_transformer_model compatible with transformers lib's model wrapper of different heads where there is self.model pointing to transformer model without heads so that code is cleaner
        # using a weak reference to avoid circular references
        # using object.setattr to bypass module registration, ensures that the referenced module is not treated as a submodule, thus avoiding issues like double-wrapping in FSDP
        object.__setattr__(self, "model", weakref.proxy(self[0].auto_model))
        self.model: PreTrainedModel

        self.config = self.model.config

        # typically loss is designed as a loss_function, but here it is a module to follow sentence transformer's design, however, to keep it compliant with transformers lib, we need to make the loss as part of the model's forward method.
        # set up loss modules that use the old interface. They receive the delegate so that when they call model.forward (or indirectly via model.decode, etc.), the vanilla forward (with the original behavior) is used.
        object.__setattr__(self, "loss_module_pairwise", MarginPairwiseLoss(model=self))
        self.loss_module_pairwise: "MarginPairwiseLoss"
        object.__setattr__(self, "loss_module_cosent", MarginCoSENTLoss(model=self))
        self.loss_module_cosent: "MarginCoSENTLoss"

    def set_decoder(self, decoder):
        self[0].auto_model = decoder
        self.model = self[0].auto_model

    def get_decoder(self):
        return self.model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end. 
        This is be compatible with CausalLM e.g. LlamaForCausalLM, Qwen2ForCausalLM etc.
        """
        self.model.resize_token_embeddings(new_num_tokens)

    def forward(self, *args, **kwargs) -> SentenceTransformerOutput:
        """
        Args:
            prompt_input_ids: Tensor of shape (batch_size, seq_length)
            prompt_attention_mask: Tensor of shape (batch_size, seq_length)
            chosen_input_ids: Tensor of shape (batch_size, seq_length)
            chosen_attention_mask: Tensor of shape (batch_size, seq_length)
            rejected_input_ids: Tensor of shape (batch_size, seq_length)
            rejected_attention_mask: Tensor of shape (batch_size, seq_length)
            label: Tensor of shape (batch_size,) representing the margin values
                (i.e. ground truth similarity difference: s(prompt,rejected) - s(prompt,chosen))

            chosen_embedding: Tensor of shape (batch_size, embedding_size)  # optional, provided to train model with doc embedding frozen
            rejected_embedding: Tensor of shape (batch_size, embedding_size)  # optional, provided to train model with doc embedding frozen

            return_verbose: bool, whether to return verbose output. Optional, defaults to False.

        Returns:
            A SentenceTransformerOutput instance.

        Or:

        Args:
            prompt_input_ids: Tensor of shape (batch_size, seq_length)
            prompt_attention_mask: Tensor of shape (batch_size, seq_length)
            chosen_input_ids: Tensor of shape (batch_size, seq_length)
            chosen_attention_mask: Tensor of shape (batch_size, seq_length)
            label: Tensor of shape (batch_size,) representing the margin values
                (i.e. ground truth similarity difference: s(prompt,rejected) - s(prompt,chosen))

            chosen_embedding: Tensor of shape (batch_size, embedding_size)  # optional, provided to train model with doc embedding frozen

            return_verbose: bool, whether to return verbose output. Optional, defaults to False.

        Returns:
            A SentenceTransformerOutput instance.

        Or:

        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            attention_mask: Tensor of shape (batch_size, seq_length)

        Returns:
            Embeddings of shape (batch_size, embedding_size) 
        """
        if (
            (len(args) >= 5 and len(args) <= 10) or
            (
                "prompt_input_ids" in kwargs and
                "prompt_attention_mask" in kwargs and
                "chosen_input_ids" in kwargs and
                "chosen_attention_mask" in kwargs and
                "label" in kwargs
            )
        ):
            return self.custom_forward(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)

    def custom_forward(self,
                prompt_input_ids: Tensor,
                prompt_attention_mask: Tensor,
                chosen_input_ids: Tensor,
                chosen_attention_mask: Tensor,
                label: Tensor,
                rejected_input_ids: Tensor = None,
                rejected_attention_mask: Tensor = None,
                chosen_embedding: Tensor = None,
                rejected_embedding: Tensor = None,
                return_verbose: bool = False,
                **kwargs
                ) -> SentenceTransformerOutput:
        """
        Args:
            prompt_input_ids: Tensor of shape (batch_size, seq_length)
            prompt_attention_mask: Tensor of shape (batch_size, seq_length)
            chosen_input_ids: Tensor of shape (batch_size, seq_length)
            chosen_attention_mask: Tensor of shape (batch_size, seq_length)
            rejected_input_ids: Tensor of shape (batch_size, seq_length)
            rejected_attention_mask: Tensor of shape (batch_size, seq_length)
            label: Tensor of shape (batch_size,) representing the margin values
                (i.e. ground truth similarity difference: s(prompt,rejected) - s(prompt,chosen))

        Returns:
            A SentenceTransformerOutput instance.
        """
        # Create dictionaries for each group: prompt (anchor), chosen (positive), rejected (negative)
        prompt_features = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
        }
        chosen_features = {
            "input_ids": chosen_input_ids,
            "attention_mask": chosen_attention_mask,
        }
        rejected_features = {
            "input_ids": rejected_input_ids,
            "attention_mask": rejected_attention_mask,
        }

        # If embeddings are provided, add them to the dictionaries
        if chosen_embedding is not None:
            chosen_features["embeddings"] = chosen_embedding
        if rejected_embedding is not None:
            rejected_features["embeddings"] = rejected_embedding

        sentence_features: List[dict[str, Tensor]] = [prompt_features, chosen_features, rejected_features]

        if rejected_input_ids is None:
            sentence_features = sentence_features[:2]
            result = self.loss_module_cosent.forward(sentence_features, label, return_verbose)
        else:
            result = self.loss_module_pairwise.forward(sentence_features, label, return_verbose)

        if return_verbose:
            loss, similarities = result
            return SentenceTransformerOutput(
                loss=loss,
                similarities=similarities
            )
        else:
            return SentenceTransformerOutput(loss=result)

    def to_string(self):
        """
        Create a stable, hashable representation of the model.
        Use only basic types from the config to avoid non-picklable objects.
        """
        # Convert the config to a dict and filter only primitive types.
        config_dict = self.config.to_dict()
        safe_config = {
            k: v for k, v in config_dict.items()
            if isinstance(v, (str, int, float, bool, type(None), list, dict))
        }
        return str(safe_config)
