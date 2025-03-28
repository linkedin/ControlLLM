"""Extension of langchain FAISS."""
from typing import Any, Iterable, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.base import AddableMixin

from controlllm.inference.llm_eval_ebr.embedding_service.embedding import EmbeddingService as Embeddings


class CustomFAISS(FAISS):
    """Extension of langchain FAISS. Current extend it to have self.embedding to recover the embedding model used"""

    def __init__(self, *args, **kwargs):
        """Initialize with necessary components."""
        super().__init__(*args, **kwargs)
        self.embedding: Embeddings = None

    # ignored the type check to be compatible with langchain's signature
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,  # type: ignore
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of unique IDs.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not isinstance(self.docstore, AddableMixin):
            raise ValueError("If trying to add texts, the underlying docstore should support " f"adding items, which {self.docstore} does not")
        # Embed and create the documents.
        # Ignored the type check as langchain's design of add_documents does not accept parameter embedding: Embeddings, so this attribute is injected
        embeddings = self.embedding.embed_documents(texts)  # type: ignore
        # todo handle embedding: [0.0] * emb_dim_size
        text_embeddings: Iterable[Tuple[str, List[float]]] = [(text, embedding) for text, embedding in zip(texts, embeddings)]
        return self.add_embeddings(text_embeddings, metadatas=metadatas, ids=ids, **kwargs)

    @classmethod
    def load_local(cls, *args: Any, **kwargs: Any) -> "CustomFAISS":
        """Load FAISS index, docstore, and index_to_docstore_id from disk.
        TODO: note that embedding attribute is not persisted in pickle file for now, so it has to be injected in addtion to load_local

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
        """
        # ignored the type check since super().load_local(*args, **kwargs) already return a instance of type CustomFAISS but annotated to FAISS
        faiss_store: CustomFAISS = super().load_local(*args, **kwargs)  # type: ignore

        # Set the embedding attribute
        faiss_store.embedding = None

        return faiss_store
