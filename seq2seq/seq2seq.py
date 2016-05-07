import keras.backend as K
from keras.engine import Input
from keras.layers import LSTM, TimeDistributed, Dense


class Seq2Seq:
    def __init__(self, encode_seq_length, decode_seq_length, n_symbols, **params):
        self.input = Input(shape=(encode_seq_length, n_symbols,))

        self.encode_seq_length = encode_seq_length
        self.decode_seq_length = decode_seq_length
        self.n_symbols = n_symbols
        self.params = params

        encoder = self.build_encoder(self.input)
        decoder = self.build_decoder(encoder)

    def get_param(self, param, default):
        if param in self.params:
            return self.params[param]
        print('Could not find param "{}" in params: Using default value {}'.format(param, default))
        return default

    def build_encoder(self, input):
        lstm = LSTM(self.get_param('n_lstm_dims', 100), return_sequences=False)(input)
        dense = Dense(self.n_symbols)(lstm)
        return dense

    def build_decoder(self, input):
        pass