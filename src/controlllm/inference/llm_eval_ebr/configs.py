import os
from pathlib import Path

# Model checkpoint path
MODEL_CHECKPOINT_PATH = "/shared/user/fine-tune/coach/model/qwen-8b-hf-checkpoint-sft-padding-semantic-cosent-450k-margin-frozen-job/checkpoint-52000"

# Content dir
CONTENT_DIR = (Path(__file__).parents[0].resolve() / "mock_data").resolve().as_posix()

# inverted_index pickle file name
EBR_INVERTED_INDEX_FILENAME = "inverted_index.pkl"

# embedding model pickle file name
EBR_MODEL_FILENAME = "embedding_model.pkl"

# default chunk size and chunk overlap
CHUNK_SIZE_LOCAL = 8192
CHUNK_OVERLAP_LOCAL = 64

# default num_processes, can be changed by os.environ["NUM_PROCESSES"] = '10'
NUM_PROCESSES = int(os.environ["NUM_PROCESSES"]) if "NUM_PROCESSES" in os.environ else 1


# type of model
SENTENCE_TRANSFORMERS = "sentence_transformers" if 'SENTENCE_TRANSFORMERS' not in os.environ else os.environ['SENTENCE_TRANSFORMERS']
# add more types of model here

MODULE = 'controlllm_daemon'

# redis configuration
REDIS_HOST = 'localhost'
REDIS_PORT = '6379'
REDIS_PASSWORD = ''
REDIS_MQ_DB = '3'
REDIS_MQ_SSL = False
REDIS_MQ_TIMEOUT = '60' if 'REDIS_MQ_TIMEOUT' not in os.environ else os.environ['REDIS_MQ_TIMEOUT']
REDIS_MQ_QUEUE = 'topic_embedding'  # topic to push and poll word/sentence embedding
REDIS_MQ_QUEUE_GEN = 'topic_embedding_generator'  # topic for text embedding generator
REDIS_MQ_QUEUE_PREDICT = 'topic_text_generator'  # topic for text model generator for prediction on text generation model
REDIS_TTL = 120
REDIS_VOCAB_BATCH_THRESHOLD = 50000 if 'REDIS_VOCAB_BATCH_THRESHOLD' not in os.environ else os.environ['REDIS_VOCAB_BATCH_THRESHOLD']
REDIS_CACHE_DB = '5'
REDIS_CACHE_SSL = False
REDIS_CACHE_DR = True  # decode response
REDIS_CACHE_EXP = 120  # cache expire time in minutes

# redis queue actions
GET_QURY_EMBEDDING = 'get_query_embedding'
GET_DOC_EMBEDDINGS = 'get_doc_embeddings'
GET_MOST_SIMILAR_DOCS = 'get_most_similar'
GET_SIMILARITY = 'get_similarity'
