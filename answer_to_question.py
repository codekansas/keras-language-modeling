from __future__ import print_function
import numpy as np

import os
from keras.engine import Input
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Activation
from keras.models import Model

# can remove this depending on ide...
os.environ['INSURANCE_QA'] = '/media/moloch/HHD/MachineLearning/data/insuranceQA/pyenc'

import sys

try:
    import cPickle as pickle
except:
    import pickle


class InsuranceQA:
    def __init__(self):
        try:
            data_path = os.environ['INSURANCE_QA']
        except KeyError:
            print("INSURANCE_QA is not set.  Set it to your clone of https://github.com/codekansas/insurance_qa_python")
            sys.exit(1)
        self.path = data_path
        self.vocab = self.load('vocabulary')
        self.table = InsuranceQA.VocabularyTable(self.vocab.values())

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    class VocabularyTable:
        ''' Identical to CharacterTable from Keras example '''
        def __init__(self, words):
            self.words = sorted(set(words))
            self.words_indices = dict((c, i) for i, c in enumerate(self.words))
            self.indices_words = dict((i, c) for i, c in enumerate(self.words))

        def encode(self, sentence, maxlen):
            indices = np.zeros((maxlen, len(self.words)))
            for i, w in enumerate(sentence):
                if i == maxlen: break
                indices[i, self.words_indices[w]] = 1
            return indices

        def decode(self, indices, calc_argmax=True, noise=0):
            if calc_argmax:
                indices = indices + np.random.rand(*indices.shape) * noise
                indices = indices.argmax(axis=-1)
            return ' '.join(self.indices_words[x] for x in indices)

def get_model(question_maxlen, answer_maxlen, vocab_len, n_hidden):
    answer = Input(shape=(answer_maxlen, vocab_len))
    # answer = Masking(mask_value=0.)(answer)

    # for i in range(2):
    #     answer = LSTM(n_hidden, return_sequences=True)(answer)

    # encoder rnn
    encode_rnn = LSTM(n_hidden, return_sequences=False)(answer)

    # repeat it maxlen times
    repeat_encoding = RepeatVector(question_maxlen)(encode_rnn)

    # decoder rnn
    decode_rnn = LSTM(n_hidden, return_sequences=True)(repeat_encoding)

    # can add more layers
    for i in range(2):
        decode_rnn = LSTM(n_hidden, return_sequences=True)(decode_rnn)

    # output
    dense = TimeDistributed(Dense(vocab_len))(decode_rnn)
    softmax = Activation('softmax')(dense)

    # compile the model
    model = Model([answer], [softmax])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    question_maxlen, answer_maxlen = 10, 40

    qa = InsuranceQA()
    batch_size = 50
    n_test = 10

    print('Generating data...')
    answers = qa.load('answers')
    questions = qa.load('train')

    def gen_questions(batch_size):
        while True:
            i = 0
            question_idx = np.zeros(shape=(batch_size, question_maxlen, len(qa.vocab)))
            answer_idx = np.zeros(shape=(batch_size, answer_maxlen, len(qa.vocab)))
            for s in questions:
                a = s['answers'][0]
                answer = qa.table.encode([qa.vocab[x] for x in answers[a]], answer_maxlen)
                question = qa.table.encode([qa.vocab[x] for x in s['question']], question_maxlen)
                answer_idx[i] = answer
                question_idx[i] = question
                i += 1
                if i == batch_size:
                    yield ([answer_idx], [question_idx])
                    i = 0

    gen = gen_questions(batch_size)
    test_gen = gen_questions(n_test)

    print('Generating model...')
    model = get_model(question_maxlen=question_maxlen, answer_maxlen=answer_maxlen,
                      vocab_len=len(qa.vocab), n_hidden=128)

    print('Training model...')
    for iteration in range(1, 200):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit_generator(gen, samples_per_epoch=100*batch_size, nb_epoch=10)

        x, y = next(test_gen)
        y = y[0]
        pred = model.predict(x, verbose=0)
        for noise in [0, 0.1, 0.2]: # not sure what noise values would be good
            print(' Noise: {}'.format(noise))
            for i in range(n_test):
                print('    Expected: {}'.format(qa.table.decode(y[i])))
                print('    Predicted: {}'.format(qa.table.decode(pred[i], noise=noise)))
