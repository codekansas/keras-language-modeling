from __future__ import print_function

import os
import random
import pickle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from scipy.stats import rankdata

random.seed(42)

data_path = '/media/moloch/HHD/MachineLearning/data/insuranceQA'
models_path = 'models/'

emb_d = pickle.load(open(os.path.join(models_path, 'word2vec.dict'), 'rb'))
rev_d = dict([(v, k) for k, v in emb_d.items()])

with open(os.path.join(data_path, 'vocabulary'), 'r') as f:
    lines = f.read()

idx_d = dict()

for line in lines.split('\n'):
    if len(line) == 0: continue
    q, a = line.split('\t')
    idx_d[q] = a


def to_idx(x):
    return int(x[4:])


def convert_from_idxs(x):
    return np.asarray([emb_d.get(idx_d[i], 0) for i in x.strip().split(' ')])


def revert(x):
    return ' '.join([rev_d.get(i, 'X') for i in x])

with open(os.path.join(data_path, 'answers.label.token_idx'), 'r') as f:
    lines = f.read()

answers = dict()

for answer in lines.split('\n'):
    if len(answer) == 0: continue
    id, txt = answer.split('\t')
    id = int(id)
    answers[id] = convert_from_idxs(txt)


def get_eval(f_name):
    with open(os.path.join(data_path, f_name), 'r') as f:
        lines = f.read()

    q_data = list()
    a_data = list()
    n_good = list()

    for qa_pair in lines.split('\n'):
        if len(qa_pair) == 0: continue

        a, q, g = qa_pair.split('\t')

        good_answers = [int(i) for i in a.strip().split(' ')]
        all_answers = [int(i) for i in g.strip().split(' ')]

        question = convert_from_idxs(q)
        q_data.append(pad_sequences([question], maxlen=maxlen_question, padding='post', truncating='post', value=0))
        a_data.append(pad_sequences([answers[i] for i in all_answers], maxlen=maxlen_answer, padding='post', truncating='post', value=0))
        n_good.append(len(good_answers))

    return q_data, a_data, n_good


def get_data(f_name):
    with open(os.path.join(data_path, f_name), 'r') as f:
        lines = f.read()

    q_data = list()
    ag_data = list()
    ab_data = list()
    targets = list()

    for qa_pair in lines.split('\n'):
        if len(qa_pair) == 0: continue

        if f_name == 'question.train.token_idx.label':
            q, a = qa_pair.split('\t')
        else:
            a, q, g = qa_pair.split('\t')

        good_answers = set([int(i) for i in a.strip().split(' ')])
        bad_answers = random.sample([int(i) for i in answers.keys() if i not in good_answers], len(good_answers))

        ag_data += [answers[int(i)] for i in good_answers]
        ab_data += [answers[int(i)] for i in bad_answers]

        question = convert_from_idxs(q)
        q_data += [question] * len(good_answers)
        targets += [0] * len(bad_answers)

    # shuffle the data (i'm not sure if keras does this, but it could help generalize)
    combined = zip(q_data, ag_data, ab_data, targets)
    random.shuffle(combined)
    q_data[:], ag_data[:], ab_data, targets[:] = zip(*combined)

    q_data = pad_sequences(q_data, maxlen=maxlen_question, padding='post', truncating='post', value=0)
    ag_data = pad_sequences(ag_data, maxlen=maxlen_answer, padding='post', truncating='post', value=0)
    ab_data = pad_sequences(ab_data, maxlen=maxlen_answer, padding='post', truncating='post', value=0)
    targets = np.asarray(targets)

    return q_data, ag_data, ab_data, targets


def get_accurate_percentage(model, questions, good_answers, bad_answers, n_eval=512):

    if n_eval != 'all':
        questions = questions[-n_eval:]
        good_answers = good_answers[-n_eval:]
        bad_answers = bad_answers[-n_eval:]

    good_output = model.predict([questions, good_answers], batch_size=128)
    bad_output = model.predict([questions, bad_answers], batch_size=128)

    correct = (good_output > bad_output).sum() / float(len(questions))

    return correct


def get_mrr(model, questions, all_answers, n_good, n_eval=-1):

    if n_eval != -1:
        questions = questions[-n_eval:]
        all_answers = all_answers[-n_eval:]
        n_good = n_good[-n_eval:]

    c = 0

    for i in range(len(questions)):
        question = questions[i]
        ans = all_answers[i]

        qs = np.repeat(question, len(ans), 0)

        sims = model.predict([qs, ans]).flatten()
        r = rankdata(sims)

        max_r = np.argmax(r)
        max_n = np.argmax(r[:n_good[i]])

        x = 1 / float(r[max_r] - r[max_n] + 1)
        c += x

        print('---------- (%d)\nQuestion:' % i, revert(question[0]))
        print('Desired answer:', revert(ans[max_n]))
        print('Highest-rank answer:', revert(ans[max_r]))
        print('Rank of best answer:', r[max_n])

    return c / len(questions)

# model parameters
n_words = 22354
maxlen_question = 10
maxlen_answer = 50

# the model being used
print('Generating model')

from keras_attention_model import make_model
train_model, test_model = make_model(maxlen_question, maxlen_answer, n_words, n_embed_dims=128, n_lstm_dims=256)

print('Getting data')
data_sets = [
    'question.train.token_idx.label',
    'question.test1.label.token_idx.pool',
    'question.test2.label.token_idx.pool',
]
q_data, ag_data, ab_data, targets = get_data(data_sets[0])
qv_data, avg_data, avb_data, v_targets = get_data(data_sets[1])

test_model.load_weights(os.path.join(models_path, 'iqa_model_for_prediction.h5'))

# found through experimentation that ~24 epochs generalized the best
print('Fitting model')
for i in range(10000):
    print('----- %d -----' % i)
    np.random.shuffle(ab_data)
    train_model.fit([q_data, ag_data, ab_data], targets, nb_epoch=1, batch_size=128, validation_data=[[qv_data, avg_data, avb_data], v_targets], shuffle=True)

    if i % 100 == 0:
        train_model.save_weights(os.path.join(models_path, 'iqa_model_for_training_iter_%d.h5' % i), overwrite=True)
        test_model.save_weights(os.path.join(models_path, 'iqa_model_for_training_iter_%d.h5' % i), overwrite=True)

train_model.save_weights(os.path.join(models_path, 'iqa_model_for_training.h5'), overwrite=True)
test_model.save_weights(os.path.join(models_path, 'iqa_model_for_prediction.h5'), overwrite=True)

test_model.load_weights(os.path.join(models_path, 'iqa_model_for_prediction.h5'))

import keras.backend as K
get_attention = K.function([test_model.layers[0].input, test_model.layers[1].input], [test_model.layers[3].get_output_at(0)])
attention = get_attention([q_data[:20], ag_data[:20]])[0]

for i in range(20):
    print('----- %d -----' % i)
    print(revert(q_data[i]))
    print(revert(ag_data[i]))
    print([np.linalg.norm(x) for x in attention[i]])

# the model actually did really well, predicted correct vs. incorrect answer 85% of the time on the validation set
test_model.load_weights(os.path.join(models_path, 'iqa_model_for_prediction.h5'))
print('Percent correct: {}'.format(get_accurate_percentage(test_model, q_data, ag_data, ab_data, n_eval='all')))

q_data, a_data, n_good = get_eval(data_sets[1])
test_model.load_weights(os.path.join(models_path, 'iqa_model_for_prediction.h5'))
print('MRR: {}'.format(get_mrr(test_model, q_data, a_data, n_good)))
