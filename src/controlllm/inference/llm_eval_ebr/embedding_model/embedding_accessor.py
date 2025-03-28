# coding: utf-8
from typing import Any, List
import psutil
import gc
import logging
# import sys
# root = os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), os.pardir), os.pardir))
# sys.path.append(root)
from langchain.embeddings.base import Embeddings
from controlllm.inference.llm_eval_ebr.configs import SENTENCE_TRANSFORMERS, MODEL_CHECKPOINT_PATH
from controlllm.inference.llm_eval_ebr.embedding_model.sentence_transformers import EmbeddingModel as SetenceTransformersModel


class SentenceEmbeddings(Embeddings):

    embedding_model = None

    def __init__(self, model_type=SENTENCE_TRANSFORMERS, model_checkpoint_path=MODEL_CHECKPOINT_PATH, num_processes=1):
        self.logger = logging.getLogger(__name__)
        self._model_type = model_type

        if self.embedding_model is None:
            if self._model_type == SENTENCE_TRANSFORMERS:
                # default to use gensim for loading embedding, set load_model_via_fasttext to True otherwise
                logging.info(f"Loading Sentence Embedding Model - {self._model_type}: {model_checkpoint_path:}, {num_processes:}")
                self.embedding_model = SetenceTransformersModel(model_checkpoint_path=model_checkpoint_path, num_processes=num_processes)
            # Add more model types to be supported here
            else:
                self.logger.warning("Skipped loading sentence embedding model because {self._model_type} is not supported, please load it manually by setting the embedding_model attribute")

    def load(self):
        self.logger.info('Loading word embedding model - {}'.format(self._model_type))
        self.logger.info(psutil.virtual_memory())
        self.embedding_model.load()
        self.logger.info('Loaded word embedding model')
        self.logger.info(psutil.virtual_memory())

    def embed_documents(self, texts: List[str]) -> Any:
        """Embed list of texts. Add prompt here if needed"""
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text: str) -> Any:
        """Embed query text. Add prompt here if needed"""
        return self.embedding_model.embed_query(text)

    def similarity(self, words):
        return self.embedding_model.similarity(words)

    def wmdistance(self, ls, rs):
        return self.embedding_model.wmdistance(ls, rs)

    def most_similar(self, words, k=10, separator=','):
        return self.embedding_model.most_similar(words, k, separator)


if __name__ == "__main__":
    references = 1
    while references > 0:
        references = gc.collect()
        print("Garbage Collection: Collected {} objects".format(references))
    print("get embeddings by sentence")
    sentenceEmbeddings = SentenceEmbeddings(model=SENTENCE_TRANSFORMERS)
    sentenceEmbeddings.load()
    print(sentenceEmbeddings.get('this is a test'))
