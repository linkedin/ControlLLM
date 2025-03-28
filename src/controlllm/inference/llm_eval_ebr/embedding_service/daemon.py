# coding: utf-8
import os
import gc
import time
import logging
import pickle

from controlllm.inference.llm_eval_ebr import configs
from controlllm.inference.llm_eval_ebr.embedding_service.message_queue import MessageQueue
from controlllm.inference.llm_eval_ebr.embedding_model.embedding_accessor import SentenceEmbeddings


class Daemon():
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def run(self):
        self.logger.debug('Connecting to message queue')
        MessageQueue.connect()
        self.logger.debug("Connected to message queue at {}".format(MessageQueue.queue))

        # Loading embeddings per process, loading model by env WORD_MODEL and SENT_MODEL, to-do loading it by env LANGUAGE
        self.logger.info('Loading Sentence Embedding Model')
        sentence_embeddings = SentenceEmbeddings(model=os.environ['SENT_MODEL']) if 'SENT_MODEL' in os.environ else SentenceEmbeddings()
        sentence_embeddings.load()
        self.logger.info('Loaded - All Embedding Models')
        self.logger.info('Listening...')
        i = 0
        while True:
            if i % 1000 == 0:
                objects = 1
                while objects > 0:
                    objects = gc.collect()
                    self.logger.debug("Garbage Collection - collected {}  objects".format(objects))
            i += 1

            message = MessageQueue.poll(timeout=1)
            if message is None:
                continue
            reply_queue = "{}:{}".format(MessageQueue.queue, message['uuid'])
            msg_type = message.get('msg_type', None)

            self.logger.info("Processing message - id: {}, msg_type: {}".format(message['uuid'], msg_type))
            start = time.time()
            docs = message.get('message', [])
            try:
                if msg_type == configs.GET_QURY_EMBEDDING:
                    response = sentence_embeddings.embed_query(docs[0])
                elif msg_type == configs.GET_DOC_EMBEDDINGS:
                    response = sentence_embeddings.embed_documents(docs)
                elif msg_type == configs.GET_MOST_SIMILAR_DOCS:
                    response = configs.most_similar(docs, message.get('top', 10))
                elif msg_type == configs.GET_SIMILARITY:
                    response = sentence_embeddings.similarity(docs)
                else:
                    self.logger.error("Wrong message type {}! Repsonse with 404".format(msg_type))
                    response = {'uuid': message['uuid'], 'status': 404}
            except Exception as e:
                self.logger.error("Error occured {}! Repsonse with 404".format(e))
                response = {'uuid': message['uuid'], 'status': 404}
            response = pickle.dumps(response)
            MessageQueue.offer_to(reply_queue, response)
            self.logger.info("Request {} {} - Finished in {}s".format(msg_type, message['uuid'], (time.time() - start)))
