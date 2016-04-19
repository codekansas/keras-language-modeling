import os
from gensim.models import Word2Vec
from keras.engine import Layer
import pickle

import keras.backend as K


class Word2VecEmbedding(Layer):
    def __init__(self, model_path, **kwargs):
        model = Word2Vec.load(model_path)
        self.W = K.variable(model.syn0)
        self.model_dims = model.syn0.shape
        super(Word2VecEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 2, 'Must provide a 2D input shape: (n_samples, input_vector)'
        return (input_shape[0], input_shape[1], self.model_dims[1])

    def call(self, x, mask=None):
        return K.gather(self.W, x)


def train_model():
    # train the word2vec model
    data_path = '/media/moloch/HHD/MachineLearning/data/insuranceQA'

    # read vocabulary and generate dictionary
    with open(os.path.join(data_path, 'vocabulary'), 'r') as f:
        lines = f.read()

    # generate dictionaries
    word2idx = dict()
    idx2word = dict()

    def to_idx(x):
        return int(x[4:])

    for vocab in lines.split('\n'):
        if len(vocab) == 0: continue
        s = vocab.split('\t')
        word2idx[s[1]] = to_idx(s[0])
        idx2word[to_idx(s[0])] = s[1]

    def convert(text):
        return [word2idx.get(i, len(word2idx)) for i in text.split(' ')]

    def revert(ids):
        return ' '.join([idx2word.get(i, 'UNKNOWN') for i in ids])

    # read answers
    with open(os.path.join(data_path, 'answers.label.token_idx'), 'r') as f:
        lines = f.read()

    answers = list()
    for answer in lines.split('\n'):
        if len(answer) == 0: continue
        id, txt = answer.split('\t')
        answers.append([idx2word[to_idx(i)] for i in txt.split(' ')])

    # read questions
    with open(os.path.join(data_path, 'question.train.token_idx.label'), 'r') as f:
        lines = f.read()

    questions = list()
    for question in lines.split('\n'):
        if len(question) == 0: continue
        q, a = question.split('\t')
        questions.append([idx2word[to_idx(i)] for i in q.split(' ')])

    sentences = questions + answers

    model = Word2Vec(sentences, size=100, min_count=1)
    model.save('word2vec.model')

if __name__ == '__main__':
    train_model()
    model = Word2Vec.load('word2vec.model')

    d = dict([(k, v.index) for k, v in model.vocab.items()])
    pickle.dump(d, open('word2vec.dict', 'wb'))

    # d = pickle.load(open('word2vec.dict', 'rb'))
    # print(sorted(list(d.items()), key=lambda a, b: b))
    #
    # Use the dictionary to convert sentences to vectors for dataset
