from __future__ import absolute_import

from keras import backend as K
from keras.layers import LSTM, activations


class AttentionLSTM(LSTM):
    def __init__(self, output_dim, attention_vec, attn_activation='tanh',
                 attn_inner_activation='tanh', single_attn=False,
                 n_attention_dim=None, **kwargs):
        self.attention_vec = attention_vec
        self.attn_activation = activations.get(attn_activation)
        self.attn_inner_activation = activations.get(attn_inner_activation)
        self.single_attention_param = single_attn
        self.n_attention_dim = output_dim if n_attention_dim is None else n_attention_dim

        super(AttentionLSTM, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        if hasattr(self.attention_vec, '_keras_shape'):
            attention_dim = self.attention_vec._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')

        self.U_a = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_a'.format(self.name))
        self.b_a = K.zeros((self.output_dim,), name='{}_b_a'.format(self.name))

        self.U_m = self.inner_init((attention_dim, self.output_dim),
                                   name='{}_U_m'.format(self.name))
        self.b_m = K.zeros((self.output_dim,), name='{}_b_m'.format(self.name))

        if self.single_attention_param:
            self.U_s = self.inner_init((self.output_dim, 1),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((1,), name='{}_b_s'.format(self.name))
        else:
            self.U_s = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_s'.format(self.name))
            self.b_s = K.zeros((self.output_dim,), name='{}_b_s'.format(self.name))

        self.trainable_weights += [self.U_a, self.U_m, self.U_s, self.b_a, self.b_m, self.b_s]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]

        m = self.attn_inner_activation(K.dot(h, self.U_a) * attention + self.b_a)
        # Intuitively it makes more sense to use a sigmoid (was getting some NaN problems
        # which I think might have been caused by the exponential function -> gradients blow up)
        s = self.attn_activation(K.dot(m, self.U_s) + self.b_s)

        if self.single_attention_param:
            h = h * K.repeat_elements(s, self.output_dim, axis=1)
        else:
            h = h * s

        return h, [h, c]

    def get_constants(self, x):
        constants = super(AttentionLSTM, self).get_constants(x)
        constants.append(K.dot(self.attention_vec, self.U_m) + self.b_m)
        return constants
