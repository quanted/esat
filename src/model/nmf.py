from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')
logging.getLogger("tensorflow").setLevel(logging.WARNING)


class NMFLayer(layers.Layer):
    def __init__(self, n_components, seed: int = None, optimization_method: str = "mu", **kwargs):
        """
        V ~= WH, constraints are factor contribution and factor concentration must be non-negative
        AND the sum of predicted concentrations must be less than or equal to total measured mass
        :param n_components:
        :param max_values:
        :param kwargs:
        """
        super(NMFLayer, self).__init__(**kwargs)
        self.n_components = n_components        # number of factors
        self.V = None                           # input data
        self.W = None                           # factor profile (constrained to 0:1)
        self.H = None                           # factor concentration
        self.seed = 42 if seed is None else seed
        self.optimization_method = optimization_method      # valid options are
        # multiplicative update 'mu' and projected gradient 'pg'

    def build(self, input_shape):
        m, n = input_shape
        self.V = keras.Input(shape=(m, n), dtype=tf.float32)

        w_init = tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=self.seed)
        self.W = tf.Variable(initial_value=w_init(shape=(m, self.n_components)), name="W", shape=(m, self.n_components),
                             dtype=tf.float32, trainable=True, constraint=keras.constraints.NonNeg())
        h_init = tf.random_uniform_initializer(minval=0.0, maxval=1.0, seed=self.seed)
        self.H = tf.Variable(initial_value=h_init(shape=(self.n_components, n)), name="H", shape=(self.n_components, n),
                             dtype=tf.float32, trainable=True, constraint=keras.constraints.NonNeg())

    @tf.function
    def call(self, data):
        """
        H and W matrices are updated according to the Lee and Seung's multiplicative update rule.
        https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
        :param data:
        :return:
        """
        wt = tf.transpose(self.W)
        new_h = tf.math.multiply(self.H,
                                 tf.math.divide_no_nan(
                                     tf.matmul(wt, data),
                                     # tf.matmul(wt, tf.matmul(self.W, self.H))
                                     tf.matmul(tf.matmul(wt, self.W), self.H)
                                 ))
        # new_h = tf.where(tf.less(new_h, 0), tf.zeros_like(new_h), new_h)
        ht = tf.transpose(new_h)
        new_w = tf.math.multiply(self.W,
                                 tf.math.divide_no_nan(
                                     tf.matmul(data, ht),
                                     # tf.matmul(tf.matmul(self.W, new_h), ht)
                                     tf.matmul(self.W, tf.matmul(new_h, ht))
                                 ))
        # new_w = tf.where(tf.less(new_w, 0), tf.zeros_like(new_w), new_w)
        return new_h, new_w


class NMF(keras.Model):
    def __init__(self, n_components, seed: int = 42, name="NMF", **kwargs):
        super(NMF, self).__init__(name=name, **kwargs)
        self.n_components = n_components
        self.layer = NMFLayer(seed=seed, n_components=n_components)

    @tf.function
    def call(self, inputs):
        data, uncertainty = inputs
        h, w = self.layer(data=data)

        # calculate modelled output
        wh = tf.matmul(w, h)
        return wh, h, w