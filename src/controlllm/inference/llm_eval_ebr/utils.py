import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from langchain.schema import Document


class Content:
    def __init__(self, **kwargs):
        """
        Initialize Content dynamically from keyword arguments.
        Keys will be transformed from e.g. "Job Title" to "job_title" so they are available as attributes.
        """
        for key, value in kwargs.items():
            normalized_key = key.lower().replace(" ", "_")
            setattr(self, normalized_key, value)


def to_langchain_docs(
    data_info: Dict[str, Content], index_fields: Optional[List[str]] = None, metadata_fields: Optional[List[str]] = None
) -> List[Document]:
    """
    Convert data to langchain Document to be compliant for indexing into vector db.

    Args:
        data_info (dict): data in dict format
        index_fields list(str): fields to be concatenated for generating embedding, typically text fields
        metadata_fields list(str): fields to put in metadata in vector database for filtering
            Note that the rest of fields other than index_fields are put into metadata fields if not specified
    Returns:
        list of langchain Document
    """
    # all fields in the first row
    first_row = next(iter(data_info.values()))
    all_fields = [field_name for field_name in first_row.__dict__.keys()]

    if index_fields is None:
        index_fields = all_fields

    langchain_docs = []
    for data_id, data in data_info.items():
        # put all values of index fields in page content for indexing
        page_content = "\n".join([getattr(data, index_field) for index_field in index_fields])

        # put all available fields except for index fields as metadata
        if metadata_fields is None:
            metadata_fields = [field_name for field_name in all_fields if field_name not in index_fields]

        # add to metadata
        metadata = {}
        for field_name in metadata_fields:
            metadata[field_name] = getattr(data, field_name, None)
        # todo avoid name conflict with defined field name in the raw input data
        metadata["__doc_id__"] = data_id

        langchain_docs.append(Document(page_content=page_content, metadata=metadata))

    return langchain_docs


def build_inverted_index(docs: List[Document], docstore_ids: List[str]) -> Dict[str, Any]:
    """
    Build inverted index of doc_id to list of docstore_id.

    Args:
        docs (list): list of langchain Document
        docstore_ids: list of docstore_id, note that langchain faiss generates the id sequentially
    Returns:
        Dict of __doc_id__ to list of (index, docstore_id)
    """
    # build inverted index of doc_id to index.
    id_to_index = []  # id_to_index: List[Tuple[str, int]]
    for index, doc in enumerate(docs):
        id_to_index.append((doc.metadata["__doc_id__"], index))

    # Build inverted index of doc_id to list of docstore_id.
    id_to_docstore_id: Dict[str, Any] = {}  # Dict[str, List[str]]
    for __doc_id__, index in id_to_index:
        if __doc_id__ not in id_to_docstore_id:
            id_to_docstore_id[__doc_id__] = []
        id_to_docstore_id[__doc_id__].append((index, docstore_ids[index]))

    return id_to_docstore_id


def load_mock_data(data_path: Optional[str] = None) -> Dict[str, Content]:
    """
    Load mock data from all JSON files in the specified folder (data_path)
    and form a dictionary mapping each unique content id to a Content instance.
    The Content instance is created generically using all the fields in the JSON file.
    """
    content_dict = {}

    if not data_path:
        # default: use the "mock_data" folder in the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "mock_data")

    logging.info(f"Loading data from {data_path}")

    if os.path.isfile(data_path) and data_path.endswith(".json"):
        # Load the single JSON file directly.
        with open(data_path, "r") as file:
            data = json.load(file)
        for content_id, content_info in data.items():
            content_dict[content_id] = Content(**content_info)
    else:
        # Assume data_path is a directory; iterate over all JSON files.
        for file_name in os.listdir(data_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(data_path, file_name)
                with open(file_path, "r") as file:
                    data = json.load(file)
                for content_id, content_info in data.items():
                    content_dict[content_id] = Content(**content_info)

    return content_dict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        elif isinstance(obj, np.float32):
            return obj.item()
        return json.JSONEncoder.default(self, obj)
