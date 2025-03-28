"""
Index data json files into vector database following this format:
    {
        "1": {
            "Title": "AGI and the Future of Work",
            "Link": "https://en.wikipedia.org/wiki/Artificial_general_intelligence",
            "Content": "Artificial general intelligence (AGI) is a type of highly autonomous artificial intelligence (AI) intended to match or surpass human capabilities across most or all economically valuable cognitive work. "
        },
        ...
    }
"""
import argparse
import glob
import logging
import os
import time
import warnings

from urllib3.exceptions import InsecureRequestWarning

from controlllm.inference.llm_eval_ebr.configs import (
    MODEL_CHECKPOINT_PATH,
    CHUNK_OVERLAP_LOCAL,
    CHUNK_SIZE_LOCAL,
    CONTENT_DIR,
    NUM_PROCESSES,
)
from controlllm.inference.llm_eval_ebr.vector_search import EBRSearch
from controlllm.inference.llm_eval_ebr.embedding_model.embedding_accessor import EmbeddingModel

parser = argparse.ArgumentParser(prog="run_index.py", description="Index text into vector database")
# Data
# Define a sentinel value to represent an unspecified argument
SENTINEL = object()
parser.add_argument("--model_checkpoint_path", metavar="MODEL_CHECKPOINT_PATH", default=MODEL_CHECKPOINT_PATH, type=str, help="Path to the model checkpoint")
parser.add_argument("--dataset_name", metavar="DATASET_NAME", default="controlllm_agent", type=str, help="name of dataset")
parser.add_argument("--data_path", metavar="data_path", default=CONTENT_DIR, type=str, help="input data file path, accept avro or json")
parser.add_argument("--file_format", metavar="FILE_FORMAT", default="json", type=str, help="accept avro or json, TODO: support avro")
parser.add_argument(
    "--persist_path", metavar="PERSIST_DIR", default=None, type=str, help="directory to persist the index, None means the same as model_checkpoint_path"
)
parser.add_argument(
    "--index_fields", metavar="INDEX_FIELDS", default=None, type=str, help="fields to be indexed in embedding - field names joined with `;`. None means all fields"
)
parser.add_argument("--metadata_fields", metavar="METADATA_FIELDS", default=None, type=list[str], help="metadata fields to be indexed in metadata")
parser.add_argument("--chunk", metavar="CHUNK", default=True, type=int, help="Chunk the content of index field or not")
parser.add_argument("--chunk_size", metavar="CHUNK_SIZE", default=SENTINEL, type=int, help="Chunk size of each chunk, by character, to be supported by token")
parser.add_argument("--chunk_overlap", metavar="CHUNK_OVERLAP", default=SENTINEL, type=int, help="Chunk overlap")
parser.add_argument(
    "--force_refresh",
    metavar="FORCE_REFRESH",
    default=None,
    type=bool,
    help="Check there is existing indexing, if force_refresh = True, do force refresh, if force_refresh = False skip indexing. If None, add to existing index.",
)
parser.add_argument(
    "--num_processes", metavar="NUM_PROCESSES", default=NUM_PROCESSES, type=int, help="Number of processes tto run indexing by embedding model"
)

args, _ = parser.parse_known_args()


def index() -> None:
    data_path = os.path.expanduser(args.data_path)

    # Check if --chunk_size was specified by the user or using default
    if args.chunk_size is not SENTINEL:
        chunk_size = args.chunk_size
    else:
        chunk_size = CHUNK_SIZE_LOCAL

    # Check if --chunk_size was specified by the user or using default
    if args.chunk_overlap is not SENTINEL:
        chunk_overlap = args.chunk_overlap
    else:
        chunk_overlap = CHUNK_OVERLAP_LOCAL

    # Create output directory if it does not exist
    if args.persist_path is None:
        args.persist_path = os.path.join(args.model_checkpoint_path, f"{args.dataset_name}-index")
    if not os.path.exists(args.persist_path):
        os.mkdir(args.persist_path)

    # initialize EBRSearch with hparams
    hparams = {
        "data_path": data_path,
        "index_fields": args.index_fields.split(";") if args.index_fields else None,
        "metadata_fields": args.metadata_fields,
        "persist_path": args.persist_path,
        "chunk": args.chunk,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "model": EmbeddingModel(model_checkpoint_path=args.model_checkpoint_path, num_processes=args.num_processes),
    }
    logging.info(f"EBRSearch is initiated by hparams: {hparams}")
    ebr_search = EBRSearch(**hparams)

    # refresh existing db if force_refresh is True
    if args.force_refresh and EBRSearch.db:
        # refresh and overwrite the index file
        logging.info(f"Refreshing exising index from {ebr_search.persist_path}.")
        EBRSearch.db = None
        EBRSearch.inverted_index = None
    elif EBRSearch.db is None:
        logging.info(f"No exising index found for {ebr_search.persist_path}. Trigger the indexing...")
    elif args.force_refresh is None:
        logging.info(f"Exising index found for {ebr_search.persist_path}. Adding to the existing indexing as force_refresh is None...")
    else:
        logging.info(f"Indexing has been skipped because of exising index from {ebr_search.persist_path}.")
        return

    # get data files to be indexed
    data_files = []
    if args.file_format == "avro":
        # data_files = tf.io.gfile.glob(os.path.join(data_path, "*.avro"))
        logging.error("Avro file format is not supported yet, indexing is aborted.")
        return
    elif args.file_format == "json":
        data_files = glob.glob(os.path.join(data_path, "*.json"))
    logging.info(f"Files to be indexed: {data_files}")

    # Disable the InsecureRequestWarning
    warnings.filterwarnings("ignore", category=InsecureRequestWarning)

    # index the files one by one
    start_time = time.perf_counter()
    for data_file in data_files:
        ebr_search.add_documents_from_file(data_file)
    end_time = time.perf_counter()

    logging.info(f"Index completed and saved to {ebr_search.persist_path}\nTime taken(seconds): {end_time - start_time} \n\n")


if __name__ == "__main__":
    """
    Code to test execution locally
    """
    # Configure the root logger to log messages with level INFO or higher
    logging.basicConfig(level=logging.INFO)
    index()
