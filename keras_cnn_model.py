from __future__ import print_function

##############
# Make model #
##############
import os

from keras.engine import Merge
from keras.layers import MaxPooling1D, Dropout, Embedding, Convolution1D, Dense, Lambda, TimeDistributed, \
    ActivityRegularization

models_path = 'models/'


def make_model(maxlen_question, maxlen_answer, n_words, n_embed_dims=128):
    from keras.layers import Input, merge
    from keras.models import Model
    import keras.backend as K

    # input
    question = Input(shape=(maxlen_question,), dtype='int32', name='question_input')
    answer_good = Input(shape=(maxlen_answer,), dtype='int32', name='answer_good_input')
    answer_bad = Input(shape=(maxlen_answer,), dtype='int32', name='answer_bad_input')

    # language model
    embedding = Embedding(n_words, n_embed_dims)

    # embedding
    q_emb = embedding(question)
    ag_emb = embedding(answer_good)
    ab_emb = embedding(answer_bad)

    # dense
    dense = TimeDistributed(Dense(200, activation='tanh'))
    q_dense = dense(q_emb)
    ag_dense = dense(ag_emb)
    ab_dense = dense(ab_emb)

    # regularlize
    q_dense = ActivityRegularization(l2=0.0001)(q_dense)
    ag_dense = ActivityRegularization(l2=0.0001)(ag_dense)
    ab_dense = ActivityRegularization(l2=0.0001)(ab_dense)

    # dropout
    q_dense = Dropout(0.25)(q_dense)
    ag_dense = Dropout(0.25)(ag_dense)
    ab_dense = Dropout(0.25)(ab_dense)

    # cnn
    cnns = [Convolution1D(filter_length=filt, nb_filter=1000, activation='relu', border_mode='same') for filt in [2, 3, 5, 7]]
    q_cnn = merge([cnn(q_dense) for cnn in cnns], mode='concat')
    ag_cnn = merge([cnn(ag_dense) for cnn in cnns], mode='concat')
    ab_cnn = merge([cnn(ab_dense) for cnn in cnns], mode='concat')

    # dropout
    q_cnn = Dropout(0.25)(q_cnn)
    ag_cnn = Dropout(0.25)(ag_cnn)
    ab_cnn = Dropout(0.25)(ab_cnn)

    # maxpooling
    # maxpool = MaxPooling1D(pool_length=2)
    maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
    q_pool = maxpool(q_cnn)
    ag_pool = maxpool(ag_cnn)
    ab_pool = maxpool(ab_cnn)

    # tanh
    tanh = Lambda(lambda x: K.tanh(x))
    q_out = tanh(q_pool)
    ag_out = tanh(ag_pool)
    ab_out = tanh(ab_pool)

    # merge together
    good_out = merge([q_out, ag_out], mode='cos', dot_axes=1)
    bad_out = merge([q_out, ab_out], mode='cos', dot_axes=1)
    target = merge([good_out, bad_out], name='target', mode=lambda x: K.maximum(1e-6, 0.009 - x[0] + x[1]), output_shape=lambda x: x[0])

    train_model = Model(input=[question, answer_good, answer_bad], output=target)
    test_model = Model(input=[question, answer_good], output=good_out)

    print('Compiling model...')

    # optimizer = RMSprop(lr=0.01, clipnorm=0.05)
    optimizer = 'adam'

    def loss(y_true, y_pred):
        return y_pred

    # unfortunately, the hinge loss approach means the "accuracy" metric isn't very valuable
    metrics = []

    train_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    test_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return train_model, test_model
