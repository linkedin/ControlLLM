# coding: utf-8

import pickle
import redis
from controlllm.inference.llm_eval_ebr import configs


class MessageQueue:
    connection = None
    queue = None

    @classmethod
    def connect(cls):
        cls.connection = redis.Redis(host=configs.REDIS_HOST, port=configs.REDIS_PORT, password=configs.REDIS_PASSWORD,
                                     db=configs.REDIS_MQ_DB, ssl=configs.REDIS_MQ_SSL)
        cls.queue = configs.REDIS_MQ_QUEUE
        return cls.connection.ping()

    @classmethod
    def poll(cls, timeout=configs.REDIS_MQ_TIMEOUT):
        queues = [cls.queue]
        reply = cls.connection.blpop(queues, timeout=timeout)
        if reply is None:
            return None
        s_queue, message = reply
        s_queue = s_queue.decode()
        json_message = pickle.loads(message)
        return json_message

    @classmethod
    def offer(cls, message):
        cls.connection.rpush(cls.queue, message)
        cls.connection.expire(cls.queue, configs.REDIS_TTL)

    @classmethod
    def offer_to(cls, reply_queue, message):
        cls.connection.rpush(reply_queue, message)
        cls.connection.expire(reply_queue, configs.REDIS_TTL)

    @classmethod
    def poll_from(cls, reply_queue, timeout=configs.REDIS_MQ_TIMEOUT):
        queues = [reply_queue]
        reply = cls.connection.blpop(queues, timeout=timeout)
        if reply is None:
            return None
        s_queue, response = reply
        s_queue = s_queue.decode()
        assert(s_queue == reply_queue)
        return pickle.loads(response)
