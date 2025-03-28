import time
import logging
from typing import Any, List

from tqdm import tqdm
from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer

from controlllm.inference.llm_eval_ebr.configs import MODEL_CHECKPOINT_PATH, NUM_PROCESSES
from controlllm.inference.batch_eval import EvaluationEngine

from controlllm.utils.loading_utils import ModelLoader
from controlllm.utils.checkpoint_converter import load_model_from_config
from controlllm.utils.custom_llama_recipes.model_checkpointing import load_sharded_model_single_gpu


class EmbeddingModel(Embeddings):
    def __init__(self, model_checkpoint_path: str = MODEL_CHECKPOINT_PATH):
        """Load LLM embedding model trained and use it for embed query and documents
        Args:
            modelId (ModelId): specify the embeddig model id that is supported in ModelId enum
            num_processes: Number of threads to use for embedding the documents
        Returns:
            Trained LLM embedding model
        """
        self.model_checkpoint_path = model_checkpoint_path

    def load(self):
        evaluation_engine = EvaluationEngine(model_checkpoint_path=self.model_checkpoint_path)
        logging.info(f"Loading model from {evaluation_engine.model_checkpoint_path}")
        if evaluation_engine.enable_fsdp:  # reuse the model loading logic for training for now
            logging.info(f"Model is loaded with FSDP")
            # Load the huggingface formated model from trained_from and then update the model with the sharded weights in output_dir and resume_checkpoint_folder
            evaluation_engine.configs.model_loading_config.pretrained_model_name_or_path = evaluation_engine.trained_from
            model_loader = ModelLoader(evaluation_engine.configs)
            model = model_loader.model
            evaluation_engine.enable_benchmark_debug = True  # this forces to run benchmark with single process without DDP as it is not supported to run DDP with FSDP loaded model for now. TODO: enable it
        else:
            logging.info(f"Model is loaded without FSDP")
            logging.info(f"Loading model from {evaluation_engine.model_checkpoint_path}")
            model = load_model_from_config(evaluation_engine.model_checkpoint_path)
            load_sharded_model_single_gpu(model, evaluation_engine.model_checkpoint_path, False)
            model = model.to(device=evaluation_engine.configs.setup_config.device, dtype=evaluation_engine.torch_dtype)
            model.config.use_cache = evaluation_engine.use_cache

        if not isinstance(model, SentenceTransformer):
            raise ValueError(f"Model should be of type SentenceTransformer, but got {type(model)}")

        self.embedding_model: SentenceTransformer = model
        self.emb_dim_size = self.embedding_model.get_sentence_embedding_dimension()

    def unload(self):
        self.embedding_model = None

    def embed_documents(self, texts: List[str]) -> Any:
        return self.embedding_model.encode(texts)

    def embed_query(self, text: str) -> Any:
        """Embed query text. Add prompt here if needed"""
        # text = "query: " + text
        return self.embedding_model.encode([text])[0]

