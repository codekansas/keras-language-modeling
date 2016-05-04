from __future__ import print_function

import os
import sys
import random
from time import strftime, gmtime

import pickle

from gensim.models import Word2Vec

from keras_models import *

random.seed(42)


def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'rb'))


def revert(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]

if __name__ == '__main__':
    try:
        data_path = os.environ['INSURANCE_QA']
    except KeyError:
        print("INSURANCE_QA is not set.  Set it to your clone of https://github.com/codekansas/insurance_qa_python")
        sys.exit(1)
        
    vocab = load(data_path, 'vocab')

    sentences = list()
    answers = load(data_path, 'answers')
    for id, txt in answers.items():
        sentences.append(revert(vocab, txt))
    for q in load(data_path, 'train'):
        sentences.append(revert(vocab, q['question']))

    model = Word2Vec(sentences, size=100, min_count=5, window=5, sg=1, iter=25)
    weights = model.syn0
    d = dict([(k, v.index) for k, v in model.vocab.items()])

    # this is the stored weights of an equivalent embedding layer
    emb = np.load('models/embedding_100_dim.h5')

    # load the vocabulary
    vocab = pickle.load(open(os.path.join(data_path, 'vocabulary'), 'rb'))

    # swap the word2vec weights with the embedded weights
    for i, w in vocab.items():
        if w not in d: continue
        emb[i, :] = weights[d[w], :]

    np.save(open('models/word2vec_100_dim.h5', 'wb'), emb)
