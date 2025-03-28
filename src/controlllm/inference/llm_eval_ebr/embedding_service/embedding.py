# coding: utf-8
import json
import pickle
from typing import Any, Callable, List
import uuid
import time
import logging
import os
from retry.api import retry_call

import concurrent.futures
from multiprocessing import Manager

from tqdm import tqdm

from controlllm.inference.llm_eval_ebr import configs
from controlllm.inference.llm_eval_ebr.embedding_service.message_queue import MessageQueue
from controlllm.inference.llm_eval_ebr.configs import MODEL_CHECKPOINT_PATH, NUM_PROCESSES
from controlllm.inference.llm_eval_ebr.embedding_model.embedding_accessor import SentenceEmbeddings


class EmbeddingService():
    def __init__(self, daemon=True, sent_model=configs.SENTENCE_TRANSFORMERS, num_processes: int = NUM_PROCESSES):
        # to-do: support different word_models for sentence encoding
        self.logger = logging.getLogger(__name__)

        if daemon is True:
            if 'DAEMON' in os.environ and os.environ['DAEMON'] == str(False):
                self.logger.warning('Daemon is set to true for embedding service but DAEMON evniron is False, start the serve with daemon = False')
                daemon = False
            else:
                self.logger.warning('Daemon is set to true for embedding service, please make sure that the embedding daemon is up and running!')
        self.daemon = daemon
        if daemon:
            self.logger.debug('Connecting to message queue')
            MessageQueue.connect()
            self.logger.debug("Connected to message queue at {}".format(MessageQueue.queue))
        else:
            # Loading embeddings per process, loading embedding_model by env SENTENCE_MODEL, to-do loading it by env LANGUAGE
            # Sentence Embedding
            logging.info(f"Loading Sentence Embedding Model - {sent_model:}")
            self.sentence_embeddings = SentenceEmbeddings(model=sent_model)
            self.sentence_embeddings.load()
            self.logger.info('Loaded - All Embedding Models')

        self.num_processes = num_processes
        self.sentence_emb_size = len(self.get_doc_embeddings(['this is test'])[0])

    def get_query_embedding(self, query):
        return self._get_response({
            'msg_type': configs.GET_QURY_EMBEDDING,
            'message': [query]
        })

    def get_doc_embeddings(self, docs):
        return self._get_response({
            'msg_type': configs.GET_DOC_EMBEDDINGS,
            'message': docs
        })

    def most_similar(self, query, top=10, separator=','):
        return self._get_response({
            'msg_type': configs.GET_MOST_SIMILAR_DOCS,
            'message': query,
            'top': top,
            'separator': separator
        })

    def similarity(self, docs):
        return self._get_response({
            'msg_type': configs.GET_SIMILARITY,
            'message': docs
        })

    def _get_response(self, message):
        if self.daemon:
            # to-do: use pickle to send request to improve performance
            if not isinstance(message['message'], str):
                message['message'] = json.loads(json.dumps(message['message']))
            message['uuid'] = str(uuid.uuid4())
            reply_queue = "{}:{}".format(MessageQueue.queue, message['uuid'])
            MessageQueue.offer(message=pickle.dumps(message))
            return MessageQueue.poll_from(reply_queue=reply_queue)
        else:
            # to-do: use multi-thread/async to improve performance
            start = time.time()
            msg_type = message.get('msg_type', None)
            docs = message.get('message', [])
            if msg_type == configs.GET_QURY_EMBEDDING:
                response = self.sentence_embeddings.embed_query(docs[0])
            elif msg_type == configs.GET_DOC_EMBEDDINGS:
                response = self.sentence_embeddings.embed_documents(docs)
            elif msg_type == configs.GET_MOST_SIMILAR_WORDS:
                response = configs.most_similar(docs, message.get('top', 10))
            elif msg_type == configs.GET_SIMILARITY:
                response = self.sentence_embeddings.similarity(docs)
            else:
                self.logger.error("Wrong message type {}! Repsonse with 404".format(msg_type))
                response = {'message': message, 'status': 404}
            self.logger.info("Request {}-{} - Finished in {}s".format(msg_type, docs, (time.time() - start)))
            return response

    def embed_documents(self, texts: List[str]) -> Any:
        """Embed list of texts. Add prompt here if needed"""
        # texts = ["passage: " + text for text in texts]
        # no need to do multi-processing when num_processes is 1
        if self.num_processes == 1:
            return self.get_doc_embeddings(texts)
        else:
            # Create a Manager object to manage a shared list of progress values
            manager = Manager()
            progress_list = manager.list([0] * self.num_processes)

            # Create a list of progress bars, one for each process
            progress_bars = [tqdm(total=len(texts) // self.num_processes, desc=f"Process-{i}") for i in range(self.num_processes)]

            # Using a ProcessPoolExecutor to process documents concurrently
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                futures = []

                for i in range(self.num_processes):
                    start_idx = i * (len(texts) // self.num_processes)
                    end_idx = (i + 1) * (len(texts) // self.num_processes) if i < self.num_processes - 1 else len(texts)

                    process_texts = texts[start_idx:end_idx]

                    # Submit each chunk of texts to the executor
                    future = executor.submit(worker, process_texts, progress_list, i, self.get_query_embedding)
                    futures.append(future)

                # Update the progress bars as tasks complete
                while any(progress_bar.n < progress_bar.total for progress_bar in progress_bars):
                    for i, progress_bar in enumerate(progress_bars):
                        progress_bar.n = progress_list[i]
                        progress_bar.refresh()
                    time.sleep(0.1)

            # Close progress bars
            for progress_bar in progress_bars:
                progress_bar.close()

            # Flatten the results into a single list
            return [result for future in futures for result in future.result()]

    def embed_query(self, text: str) -> Any:
        """Embed query text. Add prompt here if needed"""
        # text = "query: " + text
        return self.get_query_embedding(text)


def worker(texts: List[str], progress_list: Any, process_index: int, embed_fn: Callable[[str], Any], emb_dim_size: int) -> List[Any]:
    results = []
    for text in texts:
        # don't break the worker if there is any exception
        try:
            # A transient error occurred unexpectedly when a service call was made to the Gateway.
            # Retry will help to mitigate the issue
            result = retry_call(embed_fn, fargs=[text], exceptions=Exception, tries=10, delay=1, max_delay=5, backoff=2)
            results.append(result)
            progress_list[process_index] += 1
        except Exception as e:
            print(f"Error in worker {process_index}: {e}")
            results.append([0.0] * emb_dim_size)
        progress_list[process_index] += 1
    return results


if __name__ == "__main__":
    svc = EmbeddingService(daemon=False)
    print(svc.get_doc_embeddings(['this is a test']))
    print(svc.similarity('this is a test', 'this is another test'))
    query = 'this is a test'
    print(svc.get_query_embedding(query))
    print(svc.most_similar(query))
