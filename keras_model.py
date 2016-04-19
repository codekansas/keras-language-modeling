from __future__ import print_function

import os

import theano

from language_model.get_data import get_data_set

dtype = theano.config.floatX

##############
# Make model #
##############

from keras.layers import Input, LSTM, Embedding, merge
from keras.models import Model


def make_model(maxlen, n_words):
    # input
    question = Input(shape=(maxlen,), dtype='int32')
    answer = Input(shape=(maxlen,), dtype='int32')

    # language model
    embedding = Embedding(output_dim=512, input_dim=n_words, input_length=maxlen)
    f_lstm = LSTM(128)
    b_lstm = LSTM(128, go_backwards=True)

    # question part
    q_emb = embedding(question)
    q_fl = f_lstm(q_emb)
    q_bl = b_lstm(q_emb)
    q_out = merge([q_fl, q_bl], mode='concat', concat_axis=1)

    # answer part
    a_emb = embedding(answer)
    a_fl = f_lstm(a_emb)
    a_bl = b_lstm(a_emb)
    a_out = merge([a_fl, a_bl], mode='concat', concat_axis=1)

    # merge together
    target = merge([q_out, a_out], mode='cos', dot_axes=1)
    model = Model(input=[question, answer], output=target)

    print('Compiling model...')
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # get the data set
    maxlen = 200 # words
    q_train, a_train, t_train, q_test, a_test, t_test, n_words = get_data_set(maxlen)

    model = make_model(maxlen)

    print('Fitting model')
    model.fit([q_train, a_train], t_train, nb_epoch=5, batch_size=32, validation_split=0.1)
    model.save_weights('lm_weights.h5')
