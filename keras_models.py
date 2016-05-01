from __future__ import print_function

from abc import abstractmethod

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Convolution1D, Lambda, Activation, LSTM, Dense, TimeDistributed, \
    ActivityRegularization, Flatten
from keras import backend as K
from keras.models import Model

import numpy as np

from attention_lstm import AttentionLSTM


class LanguageModel:
    def __init__(self, config):
        self.question = Input(shape=(config['question_len'],), dtype='int32')
        self.answer_good = Input(shape=(config['answer_len'],), dtype='int32')
        self.answer_bad = Input(shape=(config['answer_len'],), dtype='int32')

        self.config = config
        self.model_params = config.get('model_params', dict())
        self.similarity_params = config.get('similarity_params', dict())

        self._models = None
        self._similarities = None

        self.training_model = None
        self.prediction_model = None

    def _get_inputs(self):
        return [Input(shape=(self.config['question_len'],), dtype='int32', name='question'),
                Input(shape=(self.config['answer_len'],), dtype='int32', name='answer')]

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

    def get_similarities(self):
        if self._models is None:
            self._models = self.build()
            assert len(self._models) == 2, 'build() should make question and answer language models'

        if self._similarities is None:
            question_model, answer_model = self._models

            answers_use_question = len(answer_model.internal_input_shapes) == 2

            question = question_model(self.question, self.answer_good)

            if answers_use_question:
                good = answer_model([self.question, self.answer_good])
                bad = answer_model([self.question, self.answer_bad])
            else:
                good = answer_model([self.answer_good])
                bad = answer_model([self.answer_bad])

            similarity = self.get_similarity()
            good_sim = merge([question, good], mode=similarity, output_shape=lambda x: x[:-1])
            bad_sim = merge([question, bad], mode=similarity, output_shape=lambda x: x[:-1])

            self._similarities = [good_sim, bad_sim]

        return self._similarities

    def compile(self, optimizer, **kwargs):
        similarities = self.get_similarities()

        loss = merge(similarities,
                     mode=lambda x: K.maximum(1e-6, self.config['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        self.training_model = Model(input=[self.question, self.answer_good, self.answer_bad], output=loss)
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

        self.prediction_model = Model(input=[self.question, self.answer_good], output=similarities[0])
        self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1])
        self.training_model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self.prediction_model.predict(x, **kwargs)

    def save_weights(self, file_name, **kwargs):
        assert self.training_model is not None, 'Must compile the model before saving weights'
        self.training_model.save_weights(file_name, **kwargs)

    def load_weights(self, file_name, **kwargs):
        assert self.training_model is not None, 'Must compile the model loading weights'
        self.training_model.load_weights(file_name, **kwargs)


class EmbeddingModel(LanguageModel):
    ''' This model actually performs stupidly well '''

    def build(self):
        input, _ = self._get_inputs()

        # add embedding layers
        embedding = Embedding(self.config['n_words'], self.model_params.get('n_embed_dims', 141))
        input_embedding = embedding(input)

        # dropout
        dropout = Dropout(0.5)
        input_dropout = dropout(input_embedding)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        input_pool = maxpool(input_dropout)

        # activation
        activation = Activation('tanh')
        output = activation(input_pool)

        model = Model(input=[input], output=[output])

        return model, model


class ConvolutionModel(LanguageModel):
    def mixed_filter_lengths(self, input, filter_lengths):
        cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=self.model_params.get('nb_filters', 1000),
                              activation=self.model_params.get('conv_activation', 'relu'),
                              border_mode='same') for filter_length in filter_lengths]

        return merge([cnn(input) for cnn in cnns], mode='concat'), cnns

    def build(self):
        question, answer = self._get_inputs()

        # add embedding layers
        embedding_1 = Embedding(self.config['n_words'], self.model_params.get('n_embed_dims', 100))
        embedding_2 = Embedding(self.config['n_words'], self.model_params.get('n_embed_dims', 100))
        question_embedding = embedding_1(question)
        answer_embedding = embedding_2(answer)

        # use the same word embeddings for both the question and answer models
        # embedding_1.set_weights(embedding_2.get_weights())

        # dropout
        dropout = Dropout(0.25)
        question_dropout = dropout(question_embedding)
        answer_dropout = dropout(answer_embedding)

        # # dense
        # dense_1 = TimeDistributed(Dense(self.model_params.get('n_hidden', 200), activation='tanh'))
        # dense_2 = TimeDistributed(Dense(self.model_params.get('n_hidden', 200), activation='tanh'))
        # question_dense = dense_1(question_dropout)
        # answer_dense = dense_2(answer_dropout)
        #
        # # use the same weights for both layers
        # dense_1.set_weights(dense_2.get_weights())
        #
        # question_dropout = dropout(question_dense)
        # answer_dropout = dropout(answer_dense)

        # cnn
        question_cnn, cnns_1 = self.mixed_filter_lengths(question_dropout, [2, 3, 5, 7])
        answer_cnn, cnns_2 = self.mixed_filter_lengths(answer_dropout, [2, 3, 5, 7])

        for a, b in zip(cnns_1, cnns_2):
            b.set_weights(a.get_weights())

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

        question_model = Model(input=[question], output=[question_output])
        answer_model = Model(input=[answer], output=[answer_output])
        return question_model, answer_model


class RecurrentModel(LanguageModel):
    def build(self):
        input, _ = self._get_inputs()

        # add embedding layers
        embedding = Embedding(self.config['n_words'], self.model_params.get('n_embed_dims', 141))
        input_embedding = embedding(input)

        # dropout
        dropout = Dropout(0.5)
        input_dropout = dropout(input_embedding)

        # rnn
        forward_lstm = LSTM(self.config.get('n_lstm_dims', 141), consume_less='mem', return_sequences=True)
        backward_lstm = LSTM(self.config.get('n_lstm_dims', 141), consume_less='mem', return_sequences=True)
        input_lstm = merge([forward_lstm(input_dropout), backward_lstm(input_dropout)], mode='concat', concat_axis=-1)

        # dropout
        input_dropout = dropout(input_lstm)

        # cnn
        cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=self.model_params.get('nb_filters', 1000),
                              activation=self.model_params.get('conv_activation', 'relu'),
                              border_mode='same') for filter_length in [2, 3, 5, 7]]
        input_cnn = merge([cnn(input_dropout) for cnn in cnns], mode='concat')

        # dropout
        input_dropout = dropout(input_cnn)

        # maxpooling
        maxpool = Lambda(lambda x: K.mean(K.exp(x), axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        input_pool = maxpool(input_dropout)

        # activation
        activation = Activation('tanh')
        output = activation(input_pool)

        model = Model(input=[input], output=[output])
        return model, model


class AttentionModel(LanguageModel):
    def build(self):
        question, answer = self._get_inputs()

        # add embedding layers
        embedding = Embedding(self.config['n_words'], self.model_params.get('n_embed_dims', 141))
        question_embedding = embedding(question)

        a_embedding = Embedding(self.config['n_words'], self.model_params.get('n_embed_dims', 141))
        answer_embedding = embedding(answer)

        a_embedding.set_weights(embedding.get_weights())

        # dropout
        dropout = Dropout(0.5)
        question_dropout = dropout(question_embedding)
        answer_dropout = dropout(answer_embedding)

        # rnn
        forward_lstm = LSTM(self.config.get('n_lstm_dims', 141), consume_less='mem', return_sequences=True)
        backward_lstm = LSTM(self.config.get('n_lstm_dims', 141), consume_less='mem', return_sequences=True)
        question_lstm = merge([forward_lstm(question_dropout), backward_lstm(question_dropout)], mode='concat', concat_axis=-1)

        # dropout
        question_dropout = dropout(question_lstm)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        question_pool = maxpool(question_dropout)

        # activation
        activation = Activation('tanh')
        question_output = activation(question_pool)

        question_model = Model(input=[question], output=[question_output])

        # attentional rnn
        forward_lstm = AttentionLSTM(self.config.get('n_lstm_dims', 141), question_output, consume_less='mem', return_sequences=True)
        backward_lstm = AttentionLSTM(self.config.get('n_lstm_dims', 141), question_output, consume_less='mem', return_sequences=True)
        answer_lstm = merge([forward_lstm(answer_dropout), backward_lstm(answer_dropout)], mode='concat', concat_axis=-1)

        # dropout
        answer_dropout = dropout(answer_lstm)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        answer_pool = maxpool(answer_dropout)

        # activation
        activation = Activation('tanh')
        answer_output = activation(answer_pool)

        answer_model = Model(input=[question, answer], output=[answer_output])

        return question_model, answer_model
