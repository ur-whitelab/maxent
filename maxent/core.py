import numpy as np
import tensorflow as tf
from math import sqrt
from typing import *

EPS = np.finfo(np.float32).tiny
#: Array-like type
Array = Union[np.ndarray, tf.TensorArray, float, List[float]]


class Prior:
    """Prior distribution for expected deviation from target for restraint"""

    def expected(self, l: float) -> float:
        """Expected disagreement

        :param l: The lagrange multiplier
        :return: expected disagreement
        """
        raise NotImplementedError()


class EmptyPrior(Prior):
    """No prior deviation from target for restraint (exact agreement)"""

    def expected(self, l: float):
        return 0.0


class Laplace(Prior):
    """Laplace distribution prior expected deviation from target for restraint

    :param sigma: Parameter for Laplace prior - higher means more allowable disagreement
    """

    def __init__(self, sigma: float) -> float:
        self.sigma = sigma

    def expected(self, l: float) -> float:
        return -1.0 * l * self.sigma**2 / (1.0 - l**2 * self.sigma**2 / 2)


class Restraint:
    """Restraint - includes function, target, and prior belief in deviation from target

    :param fxn: callable that returns scalar
    :param target: desired scalar value
    :param prior: prior is a :class:`Prior` for expected deviation from that target
    """

    def __init__(
        self, fxn: Callable[[Array], float], target: float, prior: Prior = EmptyPrior()
    ):
        self.target = target
        self.fxn = fxn
        self.prior = prior

    def __call__(self, traj: Array) -> float:
        return self.fxn(traj) - self.target


class _ReweightLayerLaplace(tf.keras.layers.Layer):
    """Trainable layer containing weights for maxent method"""

    def __init__(self, sigmas: Array):
        super(_ReweightLayerLaplace, self).__init__()
        l_init = tf.zeros_initializer()
        restraint_dim = len(sigmas)
        self.l = tf.Variable(
            initial_value=l_init(shape=(restraint_dim,), dtype="float32"),
            trainable=True,
            name="maxent-lambda",
            constraint=lambda x: tf.clip_by_value(
                x, -sqrt(2) / (1e-10 + sigmas), sqrt(2) / (1e-10 + sigmas)
            ),
        )
        self.sigmas = sigmas

    def call(self, gk: Array, input_weights: Array = None) -> tf.TensorArray:
        # add priors
        mask = tf.cast(tf.equal(self.sigmas, 0), tf.float32)
        two_sig = tf.math.divide_no_nan(sqrt(2), self.sigmas)
        prior_term = mask * tf.math.log(
            tf.clip_by_value(
                1.0 / (self.l + two_sig) + 1.0 / (two_sig - self.l), 1e-20, 1e8
            )
        )
        # sum-up constraint terms
        logits = tf.reduce_sum(
            -self.l[tf.newaxis, :] * gk + prior_term[tf.newaxis, :], axis=1
        )
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        if input_weights is not None:
            weights = weights * tf.reshape(input_weights, (-1,))
            weights /= tf.reduce_sum(weights)
        self.add_metric(
            tf.reduce_sum(-weights * tf.math.log(weights + EPS)),
            aggregation="mean",
            name="weight-entropy",
        )
        return weights


class _AvgLayerLaplace(tf.keras.layers.Layer):
    """Layer that returns reweighted expected value for observations"""

    def __init__(self, reweight_layer: _ReweightLayerLaplace):
        super(_AvgLayerLaplace, self).__init__()
        if type(reweight_layer) != _ReweightLayerLaplace:
            raise TypeError()
        self.rl = reweight_layer

    def call(self, gk: Array, weights: Array) -> tf.TensorArray:
        # sum over trajectories
        e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
        # add laplace term
        # cannot rely on mask due to no clip
        err_e_gk = e_gk + -1.0 * self.rl.l * self.rl.sigmas**2 / (
            1.0 - self.rl.l**2 * self.rl.sigmas**2 / 2
        )
        return err_e_gk


class _ReweightLayer(tf.keras.layers.Layer):
    """Trainable layer containing weights for maxent method"""

    def __init__(self, restraint_dim: int):
        super(_ReweightLayer, self).__init__()
        l_init = tf.zeros_initializer()
        self.l = tf.Variable(
            initial_value=l_init(shape=(restraint_dim,), dtype="float32"),
            trainable=True,
            name="maxent-lambda",
        )

    def call(self, gk: Array, input_weights: Array = None) -> tf.TensorArray:
        # sum-up constraint terms
        logits = tf.reduce_sum(-self.l[tf.newaxis, :] * gk, axis=1)
        # compute per-trajectory weights
        weights = tf.math.softmax(logits)
        if input_weights is not None:
            weights = weights * tf.reshape(input_weights, (-1,))
            weights /= tf.reduce_sum(weights)
        self.add_metric(
            tf.reduce_sum(-weights * tf.math.log(weights + 1e-30)),
            aggregation="mean",
            name="weight-entropy",
        )
        return weights


class _AvgLayer(tf.keras.layers.Layer):
    """Layer that returns reweighted expected value for observations"""

    def __init__(self, reweight_layer: _ReweightLayer):
        super(_AvgLayer, self).__init__()
        if type(reweight_layer) != _ReweightLayer:
            raise TypeError()
        self.rl = reweight_layer

    def call(self, gk: Array, weights: Array) -> tf.TensorArray:
        # sum over trajectories
        e_gk = tf.reduce_sum(gk * weights[:, tf.newaxis], axis=0)
        return e_gk


def _compute_restraints(trajs, restraints):
    N = trajs.shape[0]
    K = len(restraints)
    gk = np.empty((N, K))
    for i in range(N):
        gk[i, :] = np.array([r(trajs[i]) for r in restraints])
    return gk


class MaxentModel(tf.keras.Model):
    """Keras Maximum entropy model

    :param restraints: List of :class:`Restraint`
    :param name: Name of model
    """

    def __init__(
        self, restraints: List[Restraint], name: str = "maxent-model", **kwargs
    ):
        super(MaxentModel, self).__init__(name=name, **kwargs)
        if type(restraints) == Restraint:
            restraints = [restraints]
        self.restraints = restraints
        restraint_dim = len(restraints)
        # identify prior
        prior = type(restraints[0].prior)
        # double-check
        for r in restraints:
            if type(r.prior) != prior:
                raise ValueError("Can only do restraints of one type")
        if prior == Laplace:
            sigmas = np.array([r.prior.sigma for r in restraints], dtype=np.float32)
            self.weight_layer = _ReweightLayerLaplace(sigmas)
            self.avg_layer = _AvgLayerLaplace(self.weight_layer)
        else:
            self.weight_layer = _ReweightLayer(restraint_dim)
            self.avg_layer = _AvgLayer(self.weight_layer)
        self.lambdas = self.weight_layer.l
        self.prior = prior

    def reset_weights(self):
        """Zero out the weights of the model"""
        w = self.weight_layer.get_weights()
        self.weight_layer.set_weights(tf.zeros_like(w))

    def call(self, inputs: Union[Array, List[Array], Tuple[Array]]) -> tf.TensorArray:
        """Compute reweighted restraint values

        :param inputs: Restraint values
        :return: Weighted restraint values
        """
        input_weights = None
        if (type(inputs) == tuple or type(inputs) == list) and len(inputs) == 2:
            input_weights = inputs[1]
            inputs = inputs[0]
        weights = self.weight_layer(inputs, input_weights=input_weights)
        wgk = self.avg_layer(inputs, weights)
        return wgk

    # docstring from parent uses special sphinx stuff we cannot replicate
    def compile(
        self,
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss="mean_squared_error",
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        **kwargs
    ):
        """See ``compile`` method of  :class:`tf.keras.Model`"""
        super(MaxentModel, self).compile(
            optimizer,
            loss,
            metrics,
            loss_weights,
            weighted_metrics,
            run_eagerly,
            steps_per_execution,
            **kwargs
        )

    def fit(
        self,
        trajs: Array,
        input_weights: Array = None,
        batch_size: int = None,
        epochs: int = 128,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """Fit to given observations with restraints

        :param trajs: Observations, which can be input to :class:`Restraint`
        :param input_weights: Array of weights which will be start
        :param batch_size: Almost always should be equal to number of trajs, unless you want to mix your Lagrange multipliers across trajectories
        :param kwargs: See :class:tf.keras.Model ``fit`` method for further optional arguments, like ``verbose=0`` to hide output
        :return: The history of fit
        """

        # process kwargs
        if "verbose" not in kwargs:
            kwargs["verbose"] = 0

        gk = _compute_restraints(trajs, self.restraints)
        inputs = gk.astype(np.float32)
        if batch_size is None:
            batch_size = len(gk)
        if input_weights is None:
            input_weights = tf.ones((tf.shape(gk)[0], 1))
        result = super(MaxentModel, self).fit(
            [inputs, input_weights],
            tf.zeros_like(gk),
            batch_size=batch_size,
            epochs=epochs,
            **kwargs
        )
        self.traj_weights = self.weight_layer(inputs, input_weights)
        self.restraint_values = gk
        return result
