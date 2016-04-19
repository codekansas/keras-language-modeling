""" Some utils for training the language models on 2015's LiveQA results """

import re
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from utils.dictionary import Dictionary

rng = np.random.RandomState(42)

models_path = 'models/'
file_name = 'liveqa-2015-rels.txt'
dict_path = os.path.join(models_path, 'dict.pkl')


#################
# Download data #
#################

def download_data():
    from tqdm import tqdm
    import requests

    url = 'https://raw.githubusercontent.com/codekansas/ml/master/theano_stuff/ir/LiveQA2015-qrels-ver2.txt'
    response = requests.get(url, stream=True)

    with open(os.path.join(models_path, file_name), 'wb') as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)


#################
# Load QA pairs #
#################

def load_qa_pairs():
    if not os.path.exists(os.path.join(models_path, file_name)):
        download_data()

    with open(os.path.join(models_path, file_name), 'r') as f:
        lines = re.split('\n|\r', f.read()) # for cross-system compatibility (not sure about windows)

    questions = dict()
    answers = dict()

    qpattern = re.compile('(\d+)q\t([\w\d]+)\t\t([^\t]+)\t(.*?)\t([^\t]+)\t([^\t]+)$')
    for line in lines:
        qm = qpattern.match(line)
        if qm:
            trecid = qm.group(1)
            qid = qm.group(2)
            title = qm.group(3)
            content = qm.group(4)
            maincat = qm.group(5)
            subcat = qm.group(6)
            questions[trecid] = { 'qid': qid, 'title': title, 'content': content, 'maincat': maincat, 'subcat': subcat }
            answers[trecid] = list()
        else:
            trecid, qid, score, answer, resource = line.split('`\t`')
            trecid = trecid[:-1]
            answers[trecid].append({ 'score': score, 'answer': answer, 'resource': resource })

    assert len(questions) == len(answers) == 1087, 'There was an error processing the file somewhere (should have 1087 questions)'

    return questions, answers


#####################
# Create dictionary #
#####################

def create_dictionary_from_qas(questions=None, answers=None):

    if questions == None or answers == None:
        questions, answers = load_qa_pairs()

    if os.path.exists(dict_path):
        dic = Dictionary.load(dict_path)
    else:
        dic = Dictionary()
        for q in questions.values():
            dic.add(q['content'])
            dic.add(q['title'])

        for aa in answers.values():
            for a in aa:
                dic.add(a['answer'])
        dic.save(dict_path)

    return dic


#################
# Training set #
################

def get_data_set(maxlen, questions=None, answers=None, dic=None):

    if questions is None or answers is None:
        questions, answers = load_qa_pairs()

    if dic is None:
        dic = create_dictionary_from_qas(questions, answers)

    qs, gans, bans, targets = list(), list(), list(), list()

    for id, question in questions.items():
        qc = dic.convert(question['title'] + question['content'])[0]

        ggans = [dic.convert(a['answer'])[0] for a in answers[id] if int(a['score']) >= 3]
        bbans = [dic.convert(a['answer'])[0] for a in answers[id] if int(a['score']) < 3]

        m = min(len(ggans), len(bbans))

        qs += [qc] * m
        gans += ggans[:m]
        bans += bbans[:m]
        targets += [0] * m

    targets = np.asarray(targets)
    qs = pad_sequences(qs, maxlen=maxlen, padding='post', truncating='post', dtype='int32')
    gans = pad_sequences(gans, maxlen=maxlen, padding='post', truncating='post', dtype='int32')
    bans = pad_sequences(bans, maxlen=maxlen, padding='post', truncating='post', dtype='int32')

    return targets, qs, gans, bans, len(dic)
