from __future__ import print_function

from abc import abstractmethod

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Convolution1D, Lambda, Activation, LSTM, Dense, TimeDistributed, \
    ActivityRegularization
from keras import backend as K
from keras.models import Model

import numpy as np

from attention_lstm import AttentionLSTM


class LanguageModel:
    def __init__(self, config):
        self.question = Input(shape=(config['question_len'],), dtype='int32', name='question_base')
        self.answer_good = Input(shape=(config['answer_len'],), dtype='int32', name='answer_good_base')
        self.answer_bad = Input(shape=(config['answer_len'],), dtype='int32', name='answer_bad_base')

        self.config = config
        self.model_params = config.get('model_params', dict())
        self.similarity_params = config.get('similarity_params', dict())

        # initialize a bunch of variables that will be set later
        self._models = None
        self._similarities = None
        self._answer = None
        self._qa_model = None

        self.training_model = None
        self.prediction_model = None

    def get_answer(self):
        if self._answer is None:
            self._answer = Input(shape=(self.config['answer_len'],), dtype='int32', name='answer')
        return self._answer

    @abstractmethod
    def build(self):
        return

    def get_similarity(self):
        ''' Specify similarity in configuration under 'similarity_params' -> 'mode'
        If a parameter is needed for the model, specify it in 'similarity_params'

        Example configuration:

        config = {
            ... other parameters ...
            'similarity_params': {
                'mode': 'gesd',
                'gamma': 1,
                'c': 1,
            }
        }

        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        params = self.similarity_params
        similarity = params['mode']

        axis = lambda a: len(a._keras_shape) - 1
        dot = lambda a, b: K.batch_dot(a, b, axes=axis(a))
        l2_norm = lambda a, b: K.sqrt(K.sum((a - b) ** 2, axis=axis(a), keepdims=True))

        if similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1]))
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    def get_qa_model(self):
        if self._models is None:
            self._models = self.build()

        if self._qa_model is None:
            question_output, answer_output = self._models

            similarity = self.get_similarity()
            qa_model = merge([question_output, answer_output], mode=similarity, output_shape=lambda x: x[:-1])

            self._qa_model = Model(input=[self.question, self.get_answer()], output=[qa_model])

        return self._qa_model

    def compile(self, optimizer, **kwargs):
        qa_model = self.get_qa_model()

        good_output = qa_model([self.question, self.answer_good])
        bad_output = qa_model([self.question, self.answer_bad])

        loss = merge([good_output, bad_output],
                     mode=lambda x: K.maximum(1e-6, self.config['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        self.training_model = Model(input=[self.question, self.answer_good, self.answer_bad], output=loss)
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

        self.prediction_model = Model(input=[self.question, self.answer_good], output=good_output)
        self.prediction_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        return self.training_model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.prediction_model.predict(x, **kwargs)

    def save_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        self.prediction_model.save_weights(file_name, **kwargs)

    def load_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        self.prediction_model.load_weights(file_name, **kwargs)


class EmbeddingModel(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = self.model_params.get('initial_embed_weights', None)
        weights = weights if weights is None else [weights]
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=weights,
                              mask_zero=True)
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # dropout
        dropout = Dropout(0.5)
        question_dropout = dropout(question_embedding)
        answer_dropout = dropout(answer_embedding)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        question_maxpool = maxpool(question_dropout)
        answer_maxpool = maxpool(answer_dropout)

        # activation
        activation = Activation('tanh')
        question_output = activation(question_maxpool)
        answer_output = activation(answer_maxpool)

        return question_output, answer_output


class ConvolutionModel(LanguageModel):
    ### Validation loss at Epoch 65: 2.4e-6

    def build(self):
        assert self.config['question_len'] == self.config['answer_len']

        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = self.model_params.get('initial_embed_weights', None)
        weights = weights if weights is None else [weights]
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=weights)
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # turn off layer updating
        # embedding.params = []
        # embedding.updates = []

        # dropout
        dropout = Dropout(0.25)
        question_dropout = dropout(question_embedding)
        answer_dropout = dropout(answer_embedding)

        # dense
        dense = TimeDistributed(Dense(self.model_params.get('n_hidden', 200), activation='tanh'))
        question_dense = dense(question_dropout)
        answer_dense = dense(answer_dropout)

        # regularization
        question_dense = ActivityRegularization(l2=0.0001)(question_dense)
        answer_dense = ActivityRegularization(l2=0.0001)(answer_dense)

        # dropout
        question_dropout = dropout(question_dense)
        answer_dropout = dropout(answer_dense)

        # cnn
        cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=self.model_params.get('nb_filters', 1000),
                              activation=self.model_params.get('conv_activation', 'relu'),
                              border_mode='same') for filter_length in [2, 3, 5, 7]]
        question_cnn = merge([cnn(question_dropout) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_dropout) for cnn in cnns], mode='concat')

        # dropout
        question_dropout = dropout(question_cnn)
        answer_dropout = dropout(answer_cnn)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        question_pool = maxpool(question_dropout)
        answer_pool = maxpool(answer_dropout)

        # activation
        activation = Activation('tanh')
        question_output = activation(question_pool)
        answer_output = activation(answer_pool)

        return question_output, answer_output


class AttentionModel(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = self.model_params.get('initial_embed_weights', None)
        weights = weights if weights is None else [weights]
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=self.model_params.get('n_embed_dims', 100),
                              weights=weights,
                              mask_zero=True)
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # turn off layer updating
        # embedding.params = []
        # embedding.updates = []

        # dropout
        dropout = Dropout(0.25)
        question_dropout = dropout(question_embedding)
        answer_dropout = dropout(answer_embedding)

        # question rnn part
        f_rnn = LSTM(self.model_params.get('n_lstm_dims', 141), return_sequences=True, dropout_U=0.2, consume_less='mem')
        b_rnn = LSTM(self.model_params.get('n_lstm_dims', 141), return_sequences=True, dropout_U=0.2, consume_less='mem',
                     go_backwards=True)
        question_f_rnn = f_rnn(question_dropout)
        question_b_rnn = b_rnn(question_dropout)
        question_f_dropout = dropout(question_f_rnn)
        question_b_dropout = dropout(question_b_rnn)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        question_pool = merge([maxpool(question_f_dropout), maxpool(question_b_dropout)], mode='concat', concat_axis=-1)

        # answer rnn part
        f_rnn = AttentionLSTM(self.model_params.get('n_lstm_dims', 141), question_pool, single_attn=True, dropout_U=0.2,
                              return_sequences=True, consume_less='mem')
        b_rnn = AttentionLSTM(self.model_params.get('n_lstm_dims', 141), question_pool, single_attn=True, dropout_U=0.2,
                              return_sequences=True, consume_less='mem', go_backwards=True)
        answer_f_rnn = f_rnn(answer_dropout)
        answer_b_rnn = b_rnn(answer_dropout)
        answer_f_dropout = dropout(answer_f_rnn)
        answer_b_dropout = dropout(answer_b_rnn)
        answer_pool = merge([maxpool(answer_f_dropout), maxpool(answer_b_dropout)], mode='concat', concat_axis=-1)

        # activation
        activation = Activation('tanh')
        question_output = activation(question_pool)
        answer_output = activation(answer_pool)

        return question_output, answer_output
