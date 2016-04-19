from __future__ import print_function

try:
    import six.modes.cPickle as pickle
except ImportError:
    import pickle

import sys

from numpy import asarray
import numpy as np
from gensim.utils import tokenize


class Dictionary:
    def __init__(self):
        self._token_counts = dict()
        self._id = 0

        self.token2id = dict()
        self.id2token = list()

    def add(self, text):
        if text is None: return

        from gensim.utils import tokenize

        if isinstance(text, str):
            docs = [tokenize(text, to_lower=True)]
        else:
            docs = [tokenize(t, to_lower=True) for t in text]

        for doc in docs:
            for t in doc:
                if t in self._token_counts:
                    self._token_counts[t] += 1
                else:
                    self._token_counts[t] = 1
                    self.id2token.append(t)
                    self.token2id[t] = self._id
                    self._id += 1

    def __call__(self, item):
        return self.token2id.get(item, self._id)

    def __getitem__(self, item):
        return self.id2token[item] if 0 <= item < len(self.token2id) else 'UNKNOWN'

    def __len__(self):
        return self._id + 2

    def convert(self, text):
        if isinstance(text, str):
            docs = [tokenize(text, to_lower=True)]
        else:
            docs = [tokenize(t, to_lower=True) for t in text]

        return [asarray([self(t) for t in doc], dtype=np.int32) for doc in docs]

    def revert(self, tokens):
        texts = list()

        for token in tokens:
            texts.append(' '.join([self[t] for t in token]))

        return texts

    def strip(self, n):
        self._token_counts = dict((k, v) for k, v in self._token_counts.items() if v > n)
        self.id2token = [k for k in self._token_counts.keys()]
        self.token2id = dict((v, k) for k, v in enumerate(self.id2token))
        self._id = len(self.id2token)

    def save(self, file_name):
        pickle.dump(self, open(file_name, 'wb+'))

    def __repr__(self):
        return '<Dictionary (%d tokens)>' % self._id

    @staticmethod
    def load(file_name):
        return pickle.load(open(file_name, 'rb'))

if __name__ == '__main__':
    d = Dictionary()
    d.add('the apples and oranges are very fresh today')
    print(d)

    c = d.convert('today, i want the fresh apples and oranges')
    print(c)
    r = d.revert(c)
    print(r)
