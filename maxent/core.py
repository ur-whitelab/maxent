import numpy as np
from scipy.special import softmax
import tensorflow as tf
from math import sqrt
from .utils import merge_history
from keras import backend

EPS = np.finfo(np.float32).tiny


def traj_to_restraints(traj, inner_slice, npoints, prior, noise=0.1, time_average=7):
    '''Creates npoints restraints based on given trajectory with noise and time averaging.
    For example, it could be weekly averages with some noise.

    Returns: list of restraints, list of functions which take a matplotlib axis and lambda value and plot the restraint on it
    '''
    restraints = []
    plots = []
    # make sure it's a tuple
    inner_slice = tuple(inner_slice)
    slices = np.random.choice(
        range(0, len(traj) // time_average), replace=False, size=npoints)
    for i in slices:
        # pick random time period
        s = slice(i * time_average, i * time_average + time_average)
        v = np.clip(np.mean(traj[s], axis=0)[
                    inner_slice] + np.random.normal(scale=noise), 0, 1)

        def fxn(x, s=s, j=inner_slice):
            return tf.reduce_mean(x[s], axis=0)[j]
        print(i * time_average + time_average // 2,
              np.mean(traj[s], axis=0)[inner_slice], v)
        # need to make a multiline lambda, so fake it with tuple
        plotter = lambda ax, l, i=i, v=v, color='black', inner_slice=inner_slice, prior=prior: (
            ax.plot(i * time_average + time_average // 2,
                    v, 'o', color=color, markersize=3),
            ax.errorbar(i * time_average + time_average // 2, v, xerr=time_average //
                        2, yerr=prior.expected(float(l)), color=color, capsize=3, ms=20)
        )
        r = Restraint(fxn, v, prior)
        restraints.append(r)
        plots.append(plotter)
    return restraints, plots


class Prior:
    def expected(self, l):
        raise NotImplementedError()

    def expected_grad(self, l):
        raise NotImplementedError()

    def log_denom(self, l):
        raise NotImplementedError()


class EmptyPrior(Prior):
    def expected(self, l):
        return 0.0

    def expected_grad(self, l):
        return 0.0

    def log_denom(self, l):
        return 0.0


class Laplace(Prior):
    def __init__(self, sigma):
        self.sigma = sigma

    def expected(self, l):
        return -1. * l * self.sigma**2 / (1. - l**2 * self.sigma**2 / 2)

    def expected_grad(self, l):
        return (1.5 - 1./(l**2 * self.sigma**2))

    def log_denom(self, l):
        # cap it to stop stupid stuff
        return np.log(max(1e-8, 1. / (l + np.sqrt(2)/self.sigma) + 1. / (np.sqrt(2)/self.sigma - l)))


class Restraint:
    def __init__(self, fxn, target, prior):
        self.target = target
        self.fxn = fxn
        self.prior = prior

    def __call__(self, traj):
        return self.fxn(traj) - self.target


class AvgLayerLaplace(tf.keras.layers.Layer):
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
                             1e-8, 1e8))
        # sum-up constraint terms
        logits = tf.reduce_sum(-self.l[tf.newaxis, :]
                               * gk + prior_term[tf.newaxis, :], axis=1)
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        if input_weights is not None:
            weights = weights * tf.reshape(input_weights, (-1,))
            weights /= tf.reduce_sum(weights)
        self.add_metric(
            tf.reduce_sum(-weights * tf.math.log(weights)),
            aggregation='mean',
            name='weight-entropy')
        return weights


class AvgLayer(tf.keras.layers.Layer):
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
            tf.reduce_sum(-weights * tf.math.log(weights)),
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

class RefErrorMetric(tf.keras.metrics.Metric):
    def __init__(self, ref_traj, population_fraction=None, **kwargs):
        super(RefErrorMetric, self).__init__(name='ref-error-metric',**kwargs)
        if type(ref_traj) is np.ndarray:
            ref_traj = tf.convert_to_tensor(ref_traj, 'float32')
        self.ref_traj = ref_traj
        self.error = self.add_weight(name='ref-error', initializer='zeros')
        self.population_fraction = population_fraction
    
    def update_state(self, trajs, weights, sample_weight=None):
        mean_traj = tf.reduce_sum(
            trajs * weights[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=0)
        diff = (mean_traj - self.ref_traj)**2
        patch_mean_diff = tf.reduce_mean(tf.reduce_sum(diff * self.population_fraction[tf.newaxis, :, tf.newaxis], axis =1))
        self.error.assign(patch_mean_diff)
    def result(self):
        return self.error
    def reset_states(self):
        self.error.assign(0.)

class MaxentModel(tf.keras.Model):
    def __init__(self, restraints, use_cov=False, name='maxent-model', ref_traj=None, trajs=None, population_fraction=None, ** kwargs):
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
        if ref_traj is not None:
            self.ref_traj = ref_traj
            if population_fraction is not None:
                self.population_fraction = population_fraction
            else:
                print ('No input for population fraction. Assuming equal distribution of population in all patches.')
                self.population_fraction = 1 / \
                    ref_traj.shape[2]*np.ones(ref_traj.shape[2])
            self.traj_metric = RefErrorMetric(
                ref_traj, population_fraction=population_fraction)
        else:
            self.traj_metric = None

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
        if self.traj_metric is not None:
            self.traj_metric.update_state(
                self.trajs, weights)
            self.add_metric(
                self.traj_metric.result(),
                aggregation='mean',
                name='ref-error')
        return wgk

    def fit(self, trajs, input_weights=None, **kwargs):
        gk = _compute_restraints(trajs, self.restraints)
        inputs = gk.astype(np.float32)
        if input_weights is None:
            input_weights = tf.ones((tf.shape(gk)[0], 1))
        result = super(MaxentModel, self).fit(
            [inputs, input_weights], tf.zeros_like(gk), **kwargs)
        self.traj_weights = self.weight_layer(inputs, input_weights)
        self.restraint_values = gk
        return result


def reweight(samples, unbiased_joint, joint):
    batch_dim = samples[0].shape[0]
    logit = tf.zeros((batch_dim,))
    for i, (uj, j) in enumerate(zip(unbiased_joint, joint)):
        # reduce across other axis (summing independent variable log ps)
        logitdiff = uj.log_prob(
            samples[i] + EPS) - j.log_prob(samples[i] + EPS)
        logit += tf.reduce_sum(tf.reshape(logitdiff, (batch_dim, -1)), axis=1)
    return tf.math.softmax(logit)


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


