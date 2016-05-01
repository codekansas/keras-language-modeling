from __future__ import print_function

import os
import random
from time import strftime, gmtime

import pickle

from keras.optimizers import Adam
from scipy.stats import rankdata

from keras_models import *

random.seed(42)

data_path = '/media/moloch/HHD/MachineLearning/data/insuranceQA/pyenc'


class Evaluator:
    def __init__(self, path, conf=None):
        self.path = path
        self.conf = dict() if conf is None else conf
        self.params = conf.get('training_params', dict())
        self.answers = self.load('answers')
        self._vocab = None
        self._reverse_vocab = None
        self._eval_sets = None

    ##### Resources #####

    def load(self, name):
        return pickle.load(open(os.path.join(self.path, name), 'rb'))

    def vocab(self):
        if self._vocab is None:
            self._vocab = self.load('vocabulary')
        return self._vocab

    def reverse_vocab(self):
        if self._reverse_vocab is None:
            vocab = self.vocab()
            self._reverse_vocab = dict((v.lower(), k) for k, v in vocab.items())
        return self._reverse_vocab

    ##### Loading / saving #####

    def save_epoch(self, model, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        model.save_weights('models/weights_epoch_%d.h5' % epoch, overwrite=True)

    def load_epoch(self, model, epoch):
        assert os.path.exists('models/weights_epoch_%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
        model.load_weights('models/weights_epoch_%d.h5' % epoch)

    ##### Converting / reverting #####

    def convert(self, words):
        rvocab = self.reverse_vocab()
        if type(words) == str:
            words = words.strip().lower().split(' ')
        return [rvocab.get(w, 0) for w in words]

    def revert(self, indices):
        vocab = self.vocab()
        return [vocab.get(i, 'X') for i in indices]

    ##### Padding #####

    def padq(self, data):
        return self.pad(data, self.conf.get('question_len', None))

    def pada(self, data):
        return self.pad(data, self.conf.get('answer_len', None))

    def pad(self, data, len=None):
        from keras.preprocessing.sequence import pad_sequences
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    ##### Training #####

    def print_time(self):
        print(strftime('%Y-%m-%d %H:%M:%S :: ', gmtime()), end='')

    def train(self, model):
        eval_every = self.params.get('eval_every', None)
        save_every = self.params.get('save_every', None)
        batch_size = self.params.get('batch_size', 128)
        nb_epoch = self.params.get('nb_epoch', 10)
        split = self.params.get('validation_split', 0)

        training_set = self.load('train')

        questions = list()
        good_answers = list()

        for q in training_set:
            questions += [q['question']] * len(q['answers'])
            good_answers += [self.answers[i] for i in q['answers']]

        questions = self.padq(questions)
        good_answers = self.pada(good_answers)

        for i in range(nb_epoch):
            # bad_answers = np.roll(good_answers, random.randint(10, len(questions) - 10))
            # bad_answers = good_answers.copy()
            # random.shuffle(bad_answers)
            bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))

            print('Epoch %d :: ' % i, end='')
            self.print_time()
            model.fit([questions, good_answers, bad_answers], nb_epoch=1, batch_size=batch_size, validation_split=split)

            if eval_every is not None and (i+1) % eval_every == 0:
                self.get_mrr(model)

            if save_every is not None and (i+1) % save_every == 0:
                self.save_epoch(model, (i+1))

    ##### Evaluation #####

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
        return self._eval_sets

    def get_mrr(self, model):
        top1s = list()
        mrrs = list()

        for name, data in self.eval_sets().items():
            self.print_time()
            print('----- %s -----' % name)

            random.shuffle(data)

            if 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            c_1, c_2 = 0, 0

            for d in data:
                answers = self.pada([self.answers[i] for i in d['good'] + d['bad']])
                question = self.padq([d['question']] * len(d['good'] + d['bad']))

                n_good = len(d['good'])
                sims = model.predict([question, answers], batch_size=300).flatten()
                r = rankdata(sims, method='max')

                max_r = np.argmax(r)
                max_n = np.argmax(r[:n_good])

                c_1 += 1 if max_r == max_n else 0
                c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            top1 = c_1 / float(len(data))
            mrr = c_2 / float(len(data))

            del data

            print('Top-1 Precision: %f' % top1)
            print('MRR: %f' % mrr)

            top1s.append(top1)
            mrrs.append(mrr)

        return top1s, mrrs

if __name__ == '__main__':
    conf = {
        'question_len': 100,
        'answer_len': 100,
        'n_words': 22353, # len(vocabulary) + 1
        'margin': 0.009,

        'training_params': {
            'save_every': 1,
            # 'eval_every': 20,
            'batch_size': 128,
            'nb_epoch': 1000,
            'validation_split': 0.2,
            'optimizer': 'adam',
            # 'n_eval': 20,
        },

        'model_params': {
            'n_embed_dims': 1000,
            'n_hidden': 200,

            # convolution
            'nb_filters': 1000,
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 300,
        },

        'similarity_params': {
            'mode': 'cosine',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    evaluator = Evaluator(data_path, conf)

    ##### Define model ######
    model = ConvolutionModel(conf)
    optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer=optimizer)

    # train the model
    evaluator.train(model)

    # evaluate mrr for a particular epoch
    # evaluator.load_epoch(model, -1)
    # evaluator.get_mrr(model)
