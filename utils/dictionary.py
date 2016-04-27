from __future__ import print_function

try:
    import six.modes.cPickle as pickle
except ImportError:
    import pickle


class Dictionary:
    def __init__(self, min_len=1):
        self._token_counts = dict()
        self._id = 0
        self._min_len = min_len

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
        return self.id2token[item] if item < self._id else 'X'

    def __len__(self):
        return self._id + 1

    def convert(self, text):
        from gensim.utils import tokenize
        from numpy import asarray

        if isinstance(text, str):
            docs = [tokenize(text, to_lower=True, deacc=True)]
        else:
            docs = [tokenize(t, to_lower=True, deacc=True) for t in text]

        return [asarray([self(t) for t in doc], dtype='int32') for doc in docs]

    def revert(self, tokens):
        texts = list()

        for token in tokens:
            texts.append(' '.join([self[t] for t in token]))

        return texts

    def top(self, n):
        import operator

        sorted_tokens = sorted(self._token_counts.items(), reverse=True, key=operator.itemgetter(1))[:n]
        self._token_counts = dict((k, v) for k, v in sorted_tokens)
        self.id2token = [k for k in self._token_counts.keys()]
        self.token2id = dict((v, k) for k, v in enumerate(self.id2token))
        self._id = len(self.id2token)

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
