from gensim.models import Word2Vec
from keras.engine import Layer

import keras.backend as K


class Word2VecEmbedding(Layer):
    ''' This layer can be used instead of Keras's Embedding layer,
    if word2vec encodings are desired. The performance is generally about
    the same, although this isn't as nice of a way to do it. It uses gensim
    to train the model, and takes the path to that model in the constructor.
    '''
    def __init__(self, model_path, **kwargs):
        self.model = Word2Vec.load(model_path)
        self.W = K.variable(self.model.syn0)
        self.model_dims = self.model.syn0.shape
        super(Word2VecEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []

    def get_output_shape_for(self, input_shape):
        assert len(input_shape) == 2, 'Must provide a 2D input shape: (n_samples, input_vector)'
        return (input_shape[0], input_shape[1], self.model_dims[1])

    def call(self, x, mask=None):
        x = K.maximum(K.minimum(x, self.model_dims[1] - 1), 0)
        return K.gather(self.W, x)
