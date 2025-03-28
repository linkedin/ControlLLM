
import os
import json
from flask import Flask, jsonify, request
import sys
import logging
# root = os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), os.pardir), os.pardir))
# sys.path.append(root)
from controlllm.inference.llm_eval_ebr.embedding_service.embedding import EmbeddingService
from controlllm.inference.llm_eval_ebr.utils import NumpyEncoder

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(root)


app = Flask(__name__)
# loading embedding into memory when service starts, and load word embedding by gensim since it supports most similar and wmd
embedding_service = EmbeddingService(daemon=True, word_model='fasttext', sent_model='use', load_model_via_fasttext=False)


@app.route("/")
def ping():
    return "Welcome to embedding as a service!"


@app.route('/embedding/query/<string:query>', methods=['GET'])
def get_word_embedding(query):
    response = embedding_service.get_query_embedding(query)
    return jsonify(json.dumps(response, cls=NumpyEncoder, indent=2, ensure_ascii=False))


@app.route('/embedding/doc/<string:doc>', methods=['GET'])
def get_sentence_embedding(doc):
    response = embedding_service.get_sentence_embeddings([doc])
    return jsonify(json.dumps(response, cls=NumpyEncoder, indent=2, ensure_ascii=False))


@app.route('/most_relevant_docs/<string:query>', methods=['GET'])
def most_relevant_docs(query):
    try:
        top = int(request.args.get('top', 10))
        separator = str(request.args.get('separator', ','))
    except ValueError as err:
        top = 10
        separator = ','
        logging.error("Error occured in parsing parameters for most_similar rest service: {}".format(err))
    response = embedding_service.most_similar(query, top, separator)
    return jsonify(response)


@app.route('/similarity/<string:sentence1>/<string:sentence2>', methods=['GET'])
def similarity_between_sentences(sentence1, sentence2):
    response = embedding_service.similarity([sentence1, sentence2])
    return jsonify(json.dumps(response, cls=NumpyEncoder, indent=2, ensure_ascii=False))


def main():
    embedding_service.most_similar('this is a test', 10)  # warm up
    app.run()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    main()

# TODO: make flask service working with tornado for productive use
# from tornado.wsgi import WSGIContainer
# from tornado.ioloop import IOLoop
# from tornado.web import FallbackHandler, RequestHandler, Application

# class MainHandler(RequestHandler):
#     def get(self):
#         self.write("pong")


# flask_handler = WSGIContainer(app)

# application = Application([
#     (r"/ping", MainHandler),
#     (r".*", FallbackHandler, dict(fallback=flask_handler)),
# ])

# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.set_start_method('spawn')
#     application.listen(5000)
#     IOLoop.instance().start()
