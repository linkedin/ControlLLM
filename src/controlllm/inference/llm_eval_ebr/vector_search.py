import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from langchain.embeddings.base import Embeddings
from pydantic.v1 import BaseModel

from controlllm.inference.llm_eval_ebr.configs import CHUNK_OVERLAP_LOCAL, CHUNK_SIZE_LOCAL, EBR_INVERTED_INDEX_FILENAME, EBR_MODEL_FILENAME
from controlllm.inference.llm_eval_ebr.faiss_db import CustomFAISS as VECTOR_STORE
from controlllm.inference.llm_eval_ebr.text_splitter import RecursiveCharacterTextSplitter
from controlllm.inference.llm_eval_ebr.embedding_service.embedding import EmbeddingService
from controlllm.inference.llm_eval_ebr.utils import build_inverted_index, to_langchain_docs, load_mock_data


class EBRSearch(BaseModel):
    """
    In memory EBR by VECTOR_STORE: wrapper of langchain VECTOR_STORE

    * init_from_file: initialize the indexing from file
    * add_documents_from_file: add new documents to existing db from file
    * add_documents: add new documents from instances of given data structure
    * delete: remove documents by list of doc ids
    * knn_search: knn search from the index

    Attributes:
        db (VectorStore): vector database
        data_path (str): path to data to be index
        index_fields (list of str): fields to be concatenated for generating embedding, typically text fields
        metadata_fields (list of str): fields of metadata indexed together with embedding
        persist_path (str): path to persist the database index
        chunk (bool): do chunk the page_data or not, default No
        chunk_size (int): used only if chunk is True. Sets the maximum number of tokens per chunk
        chunk_overlap (int): used only if chunk is True
        model (str): embedding model used for generating embedding. Note that in daemon mode, it is the embedding service calling the embedding model via redis.
        inverted_index (dict of [doc_id, list of docstore_id): inverted index to find the chunked docstore_id for each doc id
    """

    # make sure db and inverted_index to be a shared reference among all instances of the EBRSearch class
    _initialized = False
    db: Optional[VECTOR_STORE] = None
    inverted_index: Optional[Dict[str, Any]] = None

    data_paths: Optional[List[str]] = []
    index_fields: Optional[List[str]] = None
    metadata_fields: Optional[List[str]] = None
    persist_path: Optional[str] = None
    chunk: Optional[bool] = True
    chunk_size: Optional[int] = CHUNK_SIZE_LOCAL
    chunk_overlap: Optional[int] = CHUNK_OVERLAP_LOCAL
    model: Embeddings = EmbeddingService()

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.persist_path is None:
            script_dir = os.path.dirname(__file__)
            self.persist_path = os.path.join(script_dir, "persistency")
        # persist the index as sub folder of the given persist_path named by indexing parameters
        if self.index_fields:
            # Initializing EBR: Generate embedding based on the index fields the user specifies
            field_suffix = "__".join(self.index_fields)
        else:
            # At inference time: Use the field specifies in `EBR_INDEXING_FIELD`
            field_suffix = "ALL"
        self.persist_path = os.path.join(
            self.persist_path, f"faiss_cs_{self.chunk_size}_co_{self.chunk_overlap}_e_{self.model.__class__.__name__}_field_{field_suffix}"
        )

        # make sure db and inverted_index are shared across instances, it is ok to have race condition as online loads the same source and read.
        if not EBRSearch._initialized:
            EBRSearch.db, EBRSearch.inverted_index = self._load()
            EBRSearch._initialized = True

    def init_from_file(self, data_path: Optional[str] = None, refresh: bool = False) -> "EBRSearch":
        """
        Initialize a vector database from raw text data in json format.
        Note that it will not index again when db has been indexed and loaded from saved index file by default.

        Args:
            data_path (str): path to raw data to create database, default to /mock_data/data.json
            fresh (bool): set it to True to force reindexing from data file
        Returns:
            A vector database indexed
        """
        if refresh and EBRSearch.db is not None:
            # refresh and overwrite the index file
            logging.info(f"Refreshing existing index from {self.persist_path}.")
            EBRSearch.db = None
            EBRSearch.inverted_index = None
            self.add_documents_from_file(data_path=data_path)
        elif EBRSearch.db is None:
            self.add_documents_from_file(data_path=data_path)
        else:
            logging.info(f"Indexing has been skipped because of existing index from {self.persist_path}.")
        return self

    def add_documents_from_file(self, data_path: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[bool]]:
        """
        Add documents to vector database from raw text data in json format and save it.

        Args:
            data_path (str): path to raw data to create database, default to /tools/mock_data/data/marketer_agent_data.json
        Returns:
            List of ids from adding the texts into the vectorstore.
            Optional[bool]: True if save is successful, False otherwise.
        """
        if data_path is not None:
            if self.data_paths is None:
                self.data_paths = []
            self.data_paths.append(data_path)
        else:
            logging.warning(
                "data path is None, continue to use default data `tools/mock_data/data/marketer_agent_data.json` to index documents from file "
            )
        logging.info(f"Loading data file {data_path}")
        data_info = load_mock_data(data_path)

        return self.add_documents(data_info)

    def add_documents(self, data_info: Dict[str, Any], **kwargs: Any) -> Tuple[Dict[str, Any], Optional[bool]]:
        """
        Add new documents to vector database and save it

        Args:
            data_info (dict): data in dict format
        Returns:
            List of ids from adding the texts into the vectorstore.
            Optional[bool]: True if save is successful, False otherwise.
        """
        logging.info("Converting to langchain doc format")
        docs = to_langchain_docs(data_info, self.index_fields, self.metadata_fields)

        if self.chunk:
            logging.info(f"Chunking by chunk_size: {self.chunk_size}, chunk_overlap: {self.chunk_overlap}")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            docs = text_splitter.split_documents(docs)

        logging.info("Indexing...")
        if EBRSearch.db is not None and EBRSearch.inverted_index is not None:
            # make sure we clean up the existing index of the same document ids before reindexing
            existing_doc_ids = set(EBRSearch.inverted_index.keys()) & set(data_info.keys())
            if len(existing_doc_ids) >= 1:
                self.delete(list(existing_doc_ids))
            ids = EBRSearch.db.add_documents(docs, **kwargs)
        else:
            logging.info("Initializing and indexing...")
            EBRSearch.db = VECTOR_STORE.from_documents(docs, self.model, **kwargs)
            EBRSearch.db.embedding = self.model
            ids = list(EBRSearch.db.index_to_docstore_id.values())

        # build inverted index from doc_id to list of docstore_id for deletion
        inverted_index = build_inverted_index(docs, ids)
        if EBRSearch.inverted_index is None:
            EBRSearch.inverted_index = inverted_index
        else:
            EBRSearch.inverted_index.update(inverted_index)

        # save all changes to local for fast reloading
        saved = self._save_local()

        return inverted_index, saved

    def delete(self, doc_ids: List[str], **kwargs: Any) -> Optional[bool]:
        """Delete by __doc_id__, in content.json, it is the do_id.

        Args:
            ids: List of doc ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise. Error if db is not initialized.
        """
        if not EBRSearch.db or not EBRSearch.inverted_index:
            raise ValueError("VECTOR_STORE db and inverted_index are not initialized")
        else:
            docstore_ids = []
            for doc_id in doc_ids:
                if doc_id not in EBRSearch.inverted_index:
                    raise ValueError(f"Document {doc_id} does not exist")
                # EBRSearch.inverted_index[doc_id]: List of Tuple(VECTOR_STORE index, VECTOR_STORE docstore_id)
                _reversed_index = {v: k for k, v in EBRSearch.db.index_to_docstore_id.items()}
                docstore_ids += [index_id_pair[1] for index_id_pair in EBRSearch.inverted_index[doc_id] if index_id_pair[1] in _reversed_index]

            # delete from db
            delete_status = False
            if len(docstore_ids) >= 1:
                delete_status = EBRSearch.db.delete(docstore_ids, **kwargs) or False
                if delete_status:
                    # delete from inverted index
                    for doc_id in doc_ids:
                        EBRSearch.inverted_index.pop(doc_id, None)
                # due to a bug in langchain, it has to be rearranged
                EBRSearch.db.index_to_docstore_id = {i: d_id for i, d_id in enumerate(EBRSearch.db.index_to_docstore_id.values())}
                save_status = self._save_local()

            # save all changes to local for fast reloading
            return bool(delete_status) & bool(save_status)

    def knn_search(
        self,
        query: str,
        score_threshold: Optional[float] = None,
        k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None,
        fetch_k: int = 200,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search the database and return the top k results based on squared l2 distance (default distance algorithm from hnswlib)

        Args:
            query (str): user's query
            threshold (float): relevance threshold (l2 distance)
            k (int): number of top nearest neighbor to fetch
            filter_criteria (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
        Returns:
            data info in dict format: List[{DOC_ID: Content, Score: score}]
        """
        logging.info(f"Use the embedding at persist_path {self.persist_path}")
        if not EBRSearch.db:
            raise ValueError("EBR db is not initialized")

        topk_doc = EBRSearch.db.similarity_search_with_relevance_scores(
            query=query, k=k, filter=filter_criteria, fetch_k=fetch_k, score_threshold=score_threshold, **kwargs
        )

        return [
            {"__doc_id__": doc.metadata["__doc_id__"], "page_data": doc.page_data, "metadata": doc.metadata, "score": score} for doc, score in topk_doc
        ]

    def _save_local(self) -> Optional[bool]:
        """Save db and inverted index to local file."""
        if self.persist_path and EBRSearch.db and EBRSearch.inverted_index:
            logging.info(f"Saving indexed db to {self.persist_path}")
            EBRSearch.db.save_local(self.persist_path)
            logging.info(f"Saving inverted index to {self.persist_path}")
            with open(f"{self.persist_path}/{EBR_INVERTED_INDEX_FILENAME}", "wb") as f:
                pickle.dump(EBRSearch.inverted_index, f)
            with open(f"{self.persist_path}/{EBR_MODEL_FILENAME}", "wb") as f:
                EBRSearch.db.embedding.unload()
                pickle.dump(EBRSearch.db.embedding, f)
            logging.info(f"Saved successfully to {self.persist_path}")
            return True
        else:
            logging.warning("Failed to save db and inverted index to local file, persist path is not specified")
            return False

    def _load(self) -> Tuple[Optional[VECTOR_STORE], Optional[Dict[str, Any]]]:
        """Load vector db from persisted index."""
        # skip indexing if the init index was done
        if self.persist_path and os.path.exists(self.persist_path) and os.path.exists(f"{self.persist_path}/{EBR_INVERTED_INDEX_FILENAME}"):
            logging.info(f"Loading database from persisted file {self.persist_path}")
            db = VECTOR_STORE.load_local(folder_path=self.persist_path, embeddings=self.model)
            with open(f"{self.persist_path}/{EBR_INVERTED_INDEX_FILENAME}", "rb") as f:
                inverted_index = pickle.load(f)
            with open(f"{self.persist_path}/{EBR_MODEL_FILENAME}", "rb") as f:
                db.embedding = pickle.load(f)
                EBRSearch.db.embedding.load_model()
            logging.info("vector db initialization successful!")
        else:
            db, inverted_index = None, None
            logging.info(f"No index is found at {self.persist_path}. Ready for indexing")

        return db, inverted_index
