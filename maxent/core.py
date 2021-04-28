import numpy as np
import tensorflow as tf
from math import sqrt

EPS = np.finfo(np.float32).tiny


class Prior:
    '''Prior distribution for expected deviation from target for restraint'''
    def expected(self, l):
        '''Returns expected disagreement'''
        raise NotImplementedError()

class EmptyPrior(Prior):
    '''No prior deviation from target for restraint (exact agreement)'''
    def expected(self, l):
        return 0.0

class Laplace(Prior):
    '''Laplace distribution prior expected deviation from target for restraint'''
    def __init__(self, sigma):
        '''Parameter for Laplace prior - higher means more allowable disagreement'''
        self.sigma = sigma

    def expected(self, l):
        return -1. * l * self.sigma**2 / (1. - l**2 * self.sigma**2 / 2)

class Restraint:
    '''Restraint - includes function, target, and prior belief in deviation from target'''
    def __init__(self, fxn, target, prior):
        '''fxn is callable that returns scalar, target is desired scalar value, and prior
        is a distribution for expected deviation from that target'''
        self.target = target
        self.fxn = fxn
        self.prior = prior

    def __call__(self, traj):
        return self.fxn(traj) - self.target


class AvgLayerLaplace(tf.keras.layers.Layer):
    '''Layer that returns reweighted expected value for observations'''
    def __init__(self, reweight_layer):
        super(AvgLayerLaplace, self).__init__()
        if type(reweight_layer) != ReweightLayerLaplace:
            raise TypeError()
        self.rl = reweight_layer

    def call(self, gk, weights):
        # sum over trajectories
        e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
        # add laplace term
        # cannot rely on mask due to no clip
        err_e_gk = e_gk + -1. * self.rl.l * self.rl.sigmas**2 / \
            (1. - self.rl.l**2 * self.rl.sigmas**2 / 2)
        return err_e_gk


class ReweightLayerLaplace(tf.keras.layers.Layer):
    '''Trainable layer containing weights for maxent method'''
    def __init__(self, sigmas):
        super(ReweightLayerLaplace, self).__init__()
        l_init = tf.random_uniform_initializer(-1, 1)
        restraint_dim = len(sigmas)
        self.l = tf.Variable(
            initial_value=l_init(shape=(restraint_dim,), dtype='float32'),
            trainable=True,
            name='maxent-lambda',
            constraint=lambda x: tf.clip_by_value(
                x, -sqrt(2) / (1e-10 + sigmas), sqrt(2) / (1e-10 + sigmas))
        )
        self.sigmas = sigmas

    def call(self, gk, input_weights=None):
        # add priors
        mask = tf.cast(tf.equal(self.sigmas, 0), tf.float32)
        two_sig = tf.math.divide_no_nan(sqrt(2), self.sigmas)
        prior_term = mask * tf.math.log(
            tf.clip_by_value(1. / (self.l + two_sig) + 1. / (two_sig - self.l),
                             1e-20, 1e8))
        # sum-up constraint terms
        logits = tf.reduce_sum(-self.l[tf.newaxis, :]
                               * gk + prior_term[tf.newaxis, :], axis=1)
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        if input_weights is not None:
            weights = weights * tf.reshape(input_weights, (-1,))
            weights /= tf.reduce_sum(weights)
        self.add_metric(
            tf.reduce_sum(-weights * tf.math.log(weights + 1e-30)),
            aggregation='mean',
            name='weight-entropy')
        return weights


class AvgLayer(tf.keras.layers.Layer):
    '''Layer that returns reweighted expected value for observations'''
    def __init__(self, reweight_layer):
        super(AvgLayer, self).__init__()
        if type(reweight_layer) != ReweightLayer:
            raise TypeError()
        self.rl = reweight_layer

    def call(self, gk, weights):
        # sum over trajectories
        e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
        return e_gk


class ReweightLayer(tf.keras.layers.Layer):
    '''Trainable layer containing weights for maxent method'''
    def __init__(self, restraint_dim):
        super(ReweightLayer, self).__init__()
        l_init = tf.zeros_initializer()
        self.l = tf.Variable(
            initial_value=l_init(shape=(restraint_dim,), dtype='float32'),
            trainable=True,
            name='maxent-lambda'
        )

    def call(self, gk, input_weights=None):
        # sum-up constraint terms
        logits = tf.reduce_sum(-self.l[tf.newaxis, :] * gk, axis=1)
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        if input_weights is not None:
            weights = weights * tf.reshape(input_weights, (-1,))
            weights /= tf.reduce_sum(weights)
        self.add_metric(
            tf.reduce_sum(-weights * tf.math.log(weights + 1e-30)),
            aggregation='mean',
            name='weight-entropy')
        return weights

def _compute_restraints(trajs, restraints):
    N = trajs.shape[0]
    K = len(restraints)
    gk = np.empty((N, K))
    for i in range(N):
        gk[i, :] = [r(trajs[i]) for r in restraints]
    return gk

class MaxentModel(tf.keras.Model):
    '''Keras Maximum entropy model'''
    def __init__(self, restraints, name='maxent-model', **kwargs):
        super(MaxentModel, self).__init__(name=name, **kwargs)
        self.restraints = restraints
        self.trajs = trajs
        restraint_dim = len(restraints)
        # identify prior
        prior = type(restraints[0].prior)
        # double-check
        for r in restraints:
            if type(r.prior) != prior:
                raise ValueError('Can only do restraints of one type')
        if prior == Laplace:
            sigmas = np.array(
                [r.prior.sigma for r in restraints], dtype=np.float32)
            self.weight_layer = ReweightLayerLaplace(sigmas)
            self.avg_layer = AvgLayerLaplace(self.weight_layer)
        else:
            self.weight_layer = ReweightLayer(restraint_dim)
            self.avg_layer = AvgLayer(self.weight_layer)
        self.lambdas = self.weight_layer.l
        self.prior = prior

    def reset_weights(self):
        w = self.weight_layer.get_weights()
        self.weight_layer.set_weights(tf.zeros_like(w))

    def call(self, inputs):
        input_weights = None
        if (type(inputs) == tuple or type(inputs) == list) and len(inputs) == 2:
            input_weights = inputs[1]
            inputs = inputs[0]
        weights = self.weight_layer(inputs, input_weights=input_weights)
        wgk = self.avg_layer(inputs, weights)
        return wgk

    def fit(self, trajs, input_weights=None, batch_size=None **kwargs):
        gk = _compute_restraints(trajs, self.restraints)
        inputs = gk.astype(np.float32)
        if batch_size is None:
            batch_size = len(gk)
        if input_weights is None:
            input_weights = tf.ones((tf.shape(gk)[0], 1))
        result = super(MaxentModel, self).fit(
            [inputs, input_weights], tf.zeros_like(gk), batch_size=batch_size, **kwargs)
        self.traj_weights = self.weight_layer(inputs, input_weights)
        self.restraint_values = gk
        return result
