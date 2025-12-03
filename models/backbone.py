import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec, Input, Dense, Dropout, BatchNormalization, \
    ReLU, Conv2D, Conv2DTranspose, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling


def SAE(dims, view, act='relu', dropout_rate=None, bn=False, dataset=None):
    # dims = [d_in, d_encoder, d_h]
    d_in, *d_interval, d_h = dims
    init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    input_name = 'v' + str(view) + '_'
    # input
    x = Input(shape=d_in, name='input' + str(view))

    # internal layers in encoder
    h = x
    for v, d in enumerate(d_interval):
        h = Dense(d, kernel_initializer=init,
                  activation=act,
                  name=input_name + 'encoder_{}'.format(v+1))(h)
        if bn:
            h = BatchNormalization()(h)
        if dropout_rate:
            h = Dropout(dropout_rate)(h)
    # hidden layer, features are extracted from here
    h = Dense(d_h, kernel_initializer=init,
              name=input_name + 'embedding')(h)

    # internal layers in decoder
    d_interval.reverse()
    y = h
    for v, d in enumerate(d_interval):
        y = Dense(d, kernel_initializer=init,
                  activation=act,
                  name=input_name + 'decoder_{}'.format(len(d_interval)-v))(y)
        if bn:
            y = BatchNormalization()(y)
        if dropout_rate:
            y = Dropout(dropout_rate)(y)
    # reconstruction
    y = Dense(d_in[0], kernel_initializer=init,
              name=input_name + 'decoder_{}'.format(0))(y)

    return Model(inputs=x, outputs=y, name=input_name + 'SAE'), \
        Model(inputs=x, outputs=h, name=input_name + 'SE')


def CAE(config, view, act='relu', dataset=None):
    d_in, *chanels, d_h = config
    kernel_size = [5, 5, 3]
    input_name = 'v' + str(view) + '_'
    # init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
    x = Input(d_in, name='input' + str(view))
    last_convshape = d_in[0]
    if last_convshape % 8 == 0:
        pad = 'same'
    else:
        pad = 'valid'
    last_convshape = int(last_convshape/8)

    # encoder
    h = x
    for v, (c, k) in enumerate(zip(chanels, kernel_size)):
        h = Conv2D(c, k, strides=2, padding='same' if v < len(chanels)-1 else pad,
                   activation=act, name=input_name+'conv_'+str(v+1))(h)
    h = Flatten(name=input_name + 'flatten')(h)

    # embedding
    h = Dense(units=d_h,
              # kernel_initializer=init,
              name=input_name+'embedding')(h)

    # decoder
    chanels.reverse()
    kernel_size.reverse()
    y = h
    y = Dense(units=chanels[0] * last_convshape * last_convshape,
              activation=act,
              # kernel_initializer=init,
              name=input_name+'dense_'+str(len(chanels)))(y)
    y = Reshape((last_convshape, last_convshape, chanels[0]),
                name=input_name+'reshape')(y)
    chanels = chanels[1:] + [d_in[2]]
    for v, (c, k) in enumerate(zip(chanels, kernel_size)):
        y = Conv2DTranspose(c, k, strides=2, padding='same' if v else pad,
                            activation=act, name=input_name+'deconv_'+str(len(chanels)-v-1))(y)
    return Model(inputs=x, outputs=y, name=input_name + 'CAE'), \
        Model(inputs=x, outputs=h, name=input_name + 'CE')


def MvAE(view_shape, hdim, dataset=None):
    aes = []
    encoders = []
    if len(view_shape[0]) == 1:
        # 1d data
        for v, sv in enumerate(view_shape):
            aev, encoderv = SAE(dims=[sv] + [500, 500, 2000] + [hdim], view=v+1,
                                dropout_rate=0, bn=False)
            aes.append(aev)
            encoders.append(encoderv)
    elif len(view_shape[0]) == 3:
        # image
        for v, sv in enumerate(view_shape):
            aev, encoderv = CAE(config=[sv] + [32, 64, 128] + [hdim], view=v+1)
            aes.append(aev)
            encoders.append(encoderv)
    return aes, encoders


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        self.clusters = None
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

