from __future__ import print_function

import os

import sys
import random
from time import strftime, gmtime

import pickle

from scipy.stats import rankdata

random.seed(42)


class Evaluator:
    def __init__(self, conf=None):
        try:
            data_path = os.environ['INSURANCE_QA']
        except KeyError:
            print("INSURANCE_QA is not set.  Set it to your clone of https://github.com/codekansas/insurance_qa_python")
            sys.exit(1)
        self.path = data_path
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
        # bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))

        val_loss = {'loss': 1., 'epoch': 0}

        for i in range(1, nb_epoch):
            # bad_answers = np.roll(good_answers, random.randint(10, len(questions) - 10))
            # bad_answers = good_answers.copy()
            # random.shuffle(bad_answers)
            bad_answers = self.pada(random.sample(self.answers.values(), len(good_answers)))

            # shuffle questions
            # zipped = zip(questions, good_answers)
            # random.shuffle(zipped)
            # questions[:], good_answers[:] = zip(*zipped)

            print('Epoch %d :: ' % i, end='')
            self.print_time()
            hist = model.fit([questions, good_answers, bad_answers], nb_epoch=1, batch_size=batch_size,
                             validation_split=split)

            if hist.history['val_loss'][0] < val_loss['loss']:
                val_loss = {'loss': hist.history['val_loss'][0], 'epoch': i}
            print('Best: Loss = {}, Epoch = {}'.format(val_loss['loss'], val_loss['epoch']))

            if eval_every is not None and i % eval_every == 0:
                self.get_mrr(model)

            if save_every is not None and i % save_every == 0:
                self.save_epoch(model, i)

    ##### Evaluation #####

    def prog_bar(self, so_far, total, n_bars=20):
        n_complete = int(so_far * n_bars / total)
        if n_complete >= n_bars - 1:
            print('\r[' + '=' * n_bars + ']', end='')
        else:
            s = '\r[' + '=' * (n_complete - 1) + '>' + '.' * (n_bars - n_complete) + ']'
            print(s, end='')

    def eval_sets(self):
        if self._eval_sets is None:
            self._eval_sets = dict([(s, self.load(s)) for s in ['dev', 'test1', 'test2']])
        return self._eval_sets

    def get_mrr(self, model, evaluate_all=False):
        top1s = list()
        mrrs = list()

        for name, data in self.eval_sets().items():
            if evaluate_all:
                self.print_time()
                print('----- %s -----' % name)

            random.shuffle(data)

            if not evaluate_all and 'n_eval' in self.params:
                data = data[:self.params['n_eval']]

            c_1, c_2 = 0, 0

            for i, d in enumerate(data):
                if evaluate_all:
                    self.prog_bar(i, len(data))

                indices = d['good'] + d['bad']
                answers = self.pada([self.answers[i] for i in indices])
                question = self.padq([d['question']] * len(indices))

                n_good = len(d['good'])
                sims = model.predict([question, answers], batch_size=500).flatten()
                r = rankdata(sims, method='max')

                max_r = np.argmax(r)
                max_n = np.argmax(r[:n_good])

                # print(' '.join(self.revert(d['question'])))
                # print(' '.join(self.revert(self.answers[indices[max_r]])))
                # print(' '.join(self.revert(self.answers[indices[max_n]])))

                c_1 += 1 if max_r == max_n else 0
                c_2 += 1 / float(r[max_r] - r[max_n] + 1)

            top1 = c_1 / float(len(data))
            mrr = c_2 / float(len(data))

            del data

            if evaluate_all:
                print('Top-1 Precision: %f' % top1)
                print('MRR: %f' % mrr)

            top1s.append(top1)
            mrrs.append(mrr)

        # rerun the evaluation if above some threshold
        if not evaluate_all:
            print('Top-1 Precision: {}'.format(top1s))
            print('MRR: {}'.format(mrrs))
            evaluate_all_threshold = self.params.get('evaluate_all_threshold', dict())
            evaluate_mode = evaluate_all_threshold.get('mode', 'all')
            mrr_theshold = evaluate_all_threshold.get('mrr', 1)
            top1_threshold = evaluate_all_threshold.get('top1', 1)

            if evaluate_mode == 'any':
                evaluate_all = evaluate_all or any([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or any([x >= mrr_theshold for x in mrrs])
            else:
                evaluate_all = evaluate_all or all([x >= top1_threshold for x in top1s])
                evaluate_all = evaluate_all or all([x >= mrr_theshold for x in mrrs])

            if evaluate_all:
                return self.get_mrr(model, evaluate_all=True)

        return top1s, mrrs


if __name__ == '__main__':
    import numpy as np

    conf = {
        'question_len': 20,
        'answer_len': 60,
        'n_words': 22353,  # len(vocabulary) + 1
        'margin': 0.05,

        'training_params': {
            'save_every': 1,
            # 'eval_every': 1,
            'batch_size': 128,
            'nb_epoch': 1000,
            'validation_split': 0.2,
            'optimizer': 'adam',
            # 'optimizer': Adam(clip_norm=0.1),
            # 'n_eval': 100,

            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
        },

        'model_params': {
            'n_embed_dims': 100,
            'n_hidden': 200,

            # convolution
            'nb_filters': 1000,  # * 4
            'conv_activation': 'relu',

            # recurrent
            'n_lstm_dims': 141,  # * 2

            'initial_embed_weights': np.load('word2vec_100_dim.embeddings'),
        },

        'similarity_params': {
            'mode': 'gesd',
            'gamma': 1,
            'c': 1,
            'd': 2,
        }
    }

    evaluator = Evaluator(conf)

    ##### Define model ######
    from seq2seq.answer_to_question import EmbeddingRNNModel
    model = EmbeddingRNNModel(conf)
    optimizer = conf.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer=optimizer)

    # save embedding layer
    # evaluator.load_epoch(model, 33)
    # embedding_layer = model.prediction_model.layers[2].layers[2]
    # evaluator.load_epoch(model, 100)
    # evaluator.train(model)
    # weights = embedding_layer.get_weights()[0]
    # np.save(open('models/embedding_1000_dim.h5', 'wb'), weights)

    # train the model
    # evaluator.load_epoch(model, 54)
    evaluator.train(model)

    # evaluate mrr for a particular epoch
    # evaluator.load_epoch(model, 54)
    # evaluator.get_mrr(model, evaluate_all=True)
