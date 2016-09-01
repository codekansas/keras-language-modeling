#!/usr/bin/env python

"""
Command-line script for generating embeddings
Useful if you want to generate larger embeddings for some models
"""

from __future__ import print_function

import os
import sys
import random
import pickle
import argparse
import logging

random.seed(42)


def load(path, name):
    return pickle.load(open(os.path.join(path, name), 'rb'))


def revert(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]

try:
    data_path = os.environ['INSURANCE_QA']
except KeyError:
    print('INSURANCE_QA is not set. Set it to your clone of https://github.com/codekansas/insurance_qa_python')
    sys.exit(1)

# parse arguments
parser = argparse.ArgumentParser(description='Generate embeddings for the InsuranceQA dataset')
parser.add_argument('--iter', metavar='N', type=int, default=10, help='number of times to run')
parser.add_argument('--size', metavar='D', type=int, default=100, help='dimensions in embedding')
args = parser.parse_args()

# configure logging
logger = logging.getLogger(os.path.basename(sys.argv[0]))
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info('running %s' % ' '.join(sys.argv))

# imports go down here because they are time-consuming
from gensim.models import Word2Vec
from keras_models import *

vocab = load(data_path, 'vocabulary')

answers = load(data_path, 'answers')
sentences = [revert(vocab, txt) for txt in answers.values()]
sentences += [revert(vocab, q['question']) for q in load(data_path, 'train')]

# run model
model = Word2Vec(sentences, size=args.size, min_count=5, window=5, sg=1, iter=args.iter)
weights = model.syn0
d = dict([(k, v.index) for k, v in model.vocab.items()])
emb = np.zeros(shape=(len(vocab)+1, args.size), dtype='float32')

for i, w in vocab.items():
    if w not in d: continue
    emb[i, :] = weights[d[w], :]

np.save(open('word2vec_%d_dim.embeddings' % args.size, 'wb'), emb)
logger.info('saved to "word2vec_%d_dim.embeddings"' % args.size)

