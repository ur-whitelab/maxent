import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .core import *
tfd = tfp.distributions
tfb = tfp.bijectors

EPS = np.finfo(np.float32).tiny


def merge_history(base, other, prefix=''):
    if base is None:
        return other
    if other is None:
        return base
    for k, v in other.history.items():
        if prefix + k in other.history:
            base.history[prefix + k].extend(v)
        else:
            base.history[prefix + k] = v
    return base


def negloglik(y, rv_y):
    logp = rv_y.log_prob(y + EPS)
    logp = tf.reduce_sum(tf.reshape(logp, (tf.shape(y)[0], -1)), axis=1)
    return -logp


def reweight(samples, unbiased_joint, joint):
    batch_dim = samples[0].shape[0]
    logit = tf.zeros((batch_dim,))
    for i, (uj, j) in enumerate(zip(unbiased_joint, joint)):
        # reduce across other axis (summing independent variable log ps)
        logitdiff = uj.log_prob(
            samples[i] + EPS) - j.log_prob(samples[i] + EPS)
        logit += tf.reduce_sum(tf.reshape(logitdiff, (batch_dim, -1)), axis=1)
    return tf.math.softmax(logit)


class ParameterJoint(tf.keras.Model):
    def __init__(self, reshapers, **kwargs):
        '''Create trainable joint model for parameters'''
        self.reshapers = reshapers
        self.output_count = len(reshapers)
        super(ParameterJoint, self).__init__(**kwargs)

    def compile(self, optimizer, **kwargs):
        if 'loss' in kwargs:
            raise ValueError('Do not set loss')
        super(ParameterJoint, self).compile(
            optimizer, loss=self.output_count * [negloglik])

    def sample(self, N, return_joint=False):
        joint = self(tf.constant([1.]))
        if type(joint) != list:
            joint = [joint]
        y = [j.sample(N) for j in joint]
        v = [self.reshapers[i](s) for i, s in enumerate(y)]
        if return_joint:
            return v, y, joint
        else:
            return v


class TrainableInputLayer(tf.keras.layers.Layer):
    ''' Create trainable input layer'''

    def __init__(self, initial_value, constraint=None, regularizer=None, **kwargs):
        super(TrainableInputLayer, self).__init__(**kwargs)
        flat = initial_value.flatten()
        self.initial_value = initial_value
        self.w = self.add_weight(
            'value',
            shape=initial_value.shape,
            initializer=tf.constant_initializer(flat),
            constraint=constraint,
            dtype=self.dtype,
            trainable=True)

    def call(self, inputs):
        batch_dim = tf.shape(inputs)[:1]
        return tf.tile(self.w[tf.newaxis, ...], tf.concat((batch_dim, tf.ones(tf.rank(self.w), dtype=tf.int32)), axis=0))


class HyperMaxentModel(MaxentModel):
    def __init__(self, restraints, prior_model, simulation, reweight=True,
                 name='hyper-maxent-model', ** kwargs):
        super(HyperMaxentModel, self).__init__(
            restraints=restraints, name=name, **kwargs)
        self.prior_model = prior_model
        self.reweight = reweight
        self.unbiased_joint = prior_model(tf.constant([1.]))
        # self.trajs = trajs
        if hasattr(self.unbiased_joint, 'sample'):
            self.unbiased_joint = [self.unbiased_joint]
        self.simulation = simulation

    def fit(self, sample_batch_size=256, final_batch_multiplier=4, param_epochs=None, outter_epochs=10, **kwargs):
        # TODO: Deal with callbacks/history
        me_history, prior_history = None, None

        # we want to reset optimizer state each time we have
        # new trajectories
        # but compile, new object assignment both
        # don't work.
        # So I guess use SGD?
        def new_optimizer():
            return self.optimizer.__class__(**self.optimizer.get_config())

        if param_epochs is None:
            param_epochs = 10
            if 'epochs' in kwargs:
                param_epochs = kwargs['epochs']
        for i in range(outter_epochs - 1):
            # sample parameters
            psample, y, joint = self.prior_model.sample(
                sample_batch_size, True)
            trajs = self.simulation(*psample)
            # get reweight, so we keep original parameter
            # probs
            rw = reweight(y, self.unbiased_joint, joint)
            # TODO reset optimizer state
            if self.reweight:
                hm = super(HyperMaxentModel, self).fit(trajs, rw, **kwargs)
            else:
                hm = super(HyperMaxentModel, self).fit(trajs, **kwargs)
            fake_x = tf.constant(sample_batch_size * [1.])
            hp = self.prior_model.fit(
                fake_x, y, sample_weight=self.traj_weights, epochs=param_epochs)
            if me_history is None:
                me_history = hm
                prior_history = hp
            else:
                me_history = merge_history(me_history, hm)
                prior_history = merge_history(prior_history, hp)

        # For final fit use more samples
        outs = []
        rws = []
        for i in range(final_batch_multiplier):
            psample, y, joint = self.prior_model.sample(
                sample_batch_size, True)
            trajs = self.simulation(*psample)
            outs.append(trajs)
            rw = reweight(y, self.unbiased_joint, joint)
            rws.append(rw)
        trajs = np.concatenate(outs, axis=0)
        rw = np.concatenate(rws, axis=0)
        self.weights_hyper = rw
        self.trajs = trajs
        # TODO reset optimizer state
        self.reset_weights()
        if self.reweight:
            hm = super(HyperMaxentModel, self).fit(trajs, rw, **kwargs)
        else:
            hm = super(HyperMaxentModel, self).fit(trajs, **kwargs)
        me_history = merge_history(me_history, hm)
        return merge_history(me_history, prior_history, 'prior-')
