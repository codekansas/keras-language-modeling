from __future__ import print_function

import os
import re
import random
import pickle
import itertools

from keras_attention_model import make_model
from utils.dictionary import Dictionary

random.seed(42)

data_path = '/media/moloch/HHD/MachineLearning/data/trecqa/jacana-qa-naacl2013-data-results'


def gen_pairs(fname, gen=False, pair_file=os.path.join('models', 'trec.pairs')):
    if os.path.exists(pair_file) and not gen:
        return pickle.load(open(pair_file, 'rb'))
    else:
        with open(os.path.join(data_path, fname), 'r') as f:
            lines = f.read()

        qa_pairs = list()

        for pair in re.finditer("<QApairs id='[\d\.]+'>(.+?)</QApairs>", lines, flags=re.DOTALL):
            text = pair.group(1)
            q = re.findall('<question>.+?</question>', text, flags=re.DOTALL)[0].split('\n')[1].split('\t')
            pos, neg = list(), list()

            for a in re.finditer("<(positive|negative)>(.+?)</.+?>", text, flags=re.DOTALL):
                cl = a.group(1)
                text = ' '.join(a.group(2).split('\n')[1].split('\t'))
                if cl[0] == 'p':
                    pos.append(text)
                else:
                    neg.append(text)

            qa_pairs.append({'question': ' '.join(q), 'positive': pos, 'negative': neg})

        pickle.dump(qa_pairs, open(pair_file, 'wb'))
        return qa_pairs


def gen_dict(pairs, gen=False, dic_file=os.path.join('models', 'trecqa.dict'), dic=None):
    if os.path.exists(dic_file) and not gen:
        dic = Dictionary.load(dic_file)
    else:
        if dic is None:
            dic = Dictionary()
        for ps in pairs:
            for q in ps:
                dic.add(q['question'])
                for ans in itertools.chain(q['positive'], q['negative']):
                    dic.add(ans)
        dic.save(dic_file)
    return dic


def gen_eval(dic, pairs, q_maxlen, a_maxlen):
    from keras.preprocessing.sequence import pad_sequences

    q_data = list()
    a_data = list()
    n_good = list()

    for pair in pairs:
        pos = dic.convert(pair['positive'])
        neg = dic.convert(pair['negative'])
        q = dic.convert(pair['question'])

        q_data.append(pad_sequences([q], maxlen=q_maxlen, padding='post', truncating='post', value=len(dic)))
        a_data.append(pad_sequences(pos + neg, maxlen=a_maxlen, padding='post', truncating='post', value=len(dic)))
        n_good.append(len(pos))

    return q_data, a_data, n_good


def gen_data(dic, pairs, q_maxlen, a_maxlen, gen=False, data_file=os.path.join('models', 'trecqa.data'), even=False):
    if os.path.exists(data_file) and not gen:
        from numpy import load as npload
        f = npload(open(data_file, 'rb'))
        return f['q'], f['p'], f['n']
    else:
        from keras.preprocessing.sequence import pad_sequences

        questions = list()
        pos_answers = list()
        neg_answers = list()

        for pair in pairs:
            pos = dic.convert(pair['positive'])
            neg = dic.convert(pair['negative'])
            q = dic.convert(pair['question'])

            questions += q * len(pos)
            pos_answers += pos
            neg_answers += neg

        questions = pad_sequences(questions, maxlen=q_maxlen, padding='post', truncating='post', dtype='int32', value=len(dic))
        pos_answers = pad_sequences(pos_answers, maxlen=a_maxlen, padding='post', truncating='post', dtype='int32', value=len(dic))
        neg_answers = pad_sequences(neg_answers, maxlen=a_maxlen, padding='post', truncating='post', dtype='int32', value=len(dic))

        if even:
            m = min(len(pos_answers), len(neg_answers))
            pos_answers = pos_answers[:m]
            neg_answers = neg_answers[:m]

        from numpy import savez as npsavez
        npsavez(open(data_file, 'wb'), q=questions, p=pos_answers, n=neg_answers)

        return questions, pos_answers, neg_answers


def get_mrr(model, questions, all_answers, n_good, n_eval=-1):
    import numpy as np
    from scipy.stats import rankdata

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

    return c / len(questions)

gen = True
qa_pairs = gen_pairs('train2393.cleanup.xml', gen=gen, pair_file=os.path.join('models', 'trecqa.pairs'))
qa_pairs_dev = gen_pairs('dev-less-than-40.manual-edit.xml', gen=gen, pair_file=os.path.join('models', 'trec_dev.pairs'))
qa_pairs_test = gen_pairs('test-less-than-40.manual-edit.xml', gen=gen, pair_file=os.path.join('models', 'trec_test.pairs'))

dic = gen_dict([qa_pairs, qa_pairs_test, qa_pairs_dev], gen=gen)

q_maxlen = 10
a_maxlen = 40
dic.top(20000)
n_words = len(dic) + 1

questions, pos_answers, neg_answers = gen_data(dic, qa_pairs, q_maxlen, a_maxlen, data_file=os.path.join('models', 'trecqa.data'), gen=gen)
questions_dev, pos_answers_dev, neg_answers_dev = gen_data(dic, qa_pairs_dev, q_maxlen, a_maxlen, even=True, data_file=os.path.join('models', 'trecqa_dev.data'), gen=gen)
questions_test, answers_test, n_good_test = gen_eval(dic, qa_pairs_test, q_maxlen, a_maxlen)

from numpy import asarray
targets = asarray([0] * len(questions))
targets_dev = asarray([0] * len(questions_dev))

print('Generating model')
train_model, test_model = make_model(q_maxlen, a_maxlen, n_words, n_embed_dims=400, n_lstm_dims=64)

for i in range(1000):
    print('----- %d -----' % i)
    import numpy.random as nprandom
    nprandom.shuffle(neg_answers)
    neg_answers_train = neg_answers[:len(pos_answers)]

    train_model.fit([questions, pos_answers, neg_answers_train], targets, nb_epoch=1, batch_size=128, validation_data=[[questions_dev, pos_answers_dev, neg_answers_dev], targets_dev])

    if i % 20 == 0:
        train_model.save_weights(os.path.join('models', 'trecqa_model_for_training_iter_%d.h5' % (i+1)), overwrite=True)
        test_model.save_weights(os.path.join('models', 'trecqa_model_for_testing_iter_%d.h5' % (i+1)), overwrite=True)
        print('MRR: {}'.format(get_mrr(test_model, questions_test, answers_test, n_good_test)))
