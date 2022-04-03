import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .core import *

tfd = tfp.distributions
tfb = tfp.bijectors

EPS = np.finfo(np.float32).tiny


def _merge_history(base, other, prefix=""):
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


def negloglik(y: Array, rv_y: tfd.Distribution) -> Array:
    """
    negative log likelihood

    :param y: observations
    :param rv_y: distribution
    :return: negative log likelihood of y
    """
    logp = rv_y.log_prob(y + EPS)
    logp = tf.reduce_sum(tf.reshape(logp, (tf.shape(y)[0], -1)), axis=1)
    return -logp


class ParameterJoint(tf.keras.Model):
    """Prior parameter model joint distribution

    This packages up how you want to sample prior paramters into one joint distribution.
    This has an important ability of reshaping output from distributions in case your simulation requires
    matrices, applying constraints, or projections.

    :param inputs: :class:`tf.keras.Input` or tuple of them.
    :param outputs: list of :py:class:`tfp.distributions.Distribution`
    :param reshapers: optional list of callables that will be called on outputs from your distribution
    """

    def __init__(
        self,
        reshapers: List[Callable[[Array], Array]] = None,
        inputs: Union[tf.keras.Input, Tuple[tf.keras.Input]] = None,
        outputs: List[tfd.Distribution] = None,
        **kwargs
    ):
        if inputs is None or outputs is None:
            raise ValueError("Must pass inputs and outputs to construct model")
        if reshapers:
            self.reshapers = reshapers
            self.output_count = len(reshapers)
        else:
            self.output_count = len(outputs)
            self.reshapers = [lambda x: x for _ in range(self.output_count)]
        super(ParameterJoint, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

    def compile(self, optimizer: object, **kwargs):
        """See ``compile`` method of  :class:`tf.keras.Model`"""
        if "loss" in kwargs:
            raise ValueError("Do not set loss")
        super(ParameterJoint, self).compile(
            optimizer, loss=self.output_count * [negloglik]
        )

    def sample(
        self, N: int, return_joint: bool = False
    ) -> Union[Tuple[Array, Array, Any], Array]:
        """Generate sample

        :param N: Number of samples (events)
        :param return_joint: return a joint :py:class:`tfp.distributions.Distribution` that can be called on ``y``
        :return: the reshaped output samples and (if ``return_joint``) a value ``y`` which can be used to compute probabilities and :py:class:`tfp.distributions.Distribution` joint
        """
        joint = self(tf.constant([1.0]))
        if type(joint) != list:
            joint = [joint]
        y = [j.sample(N) for j in joint]
        v = [self.reshapers[i](s) for i, s in enumerate(y)]
        if return_joint:
            return v, y, joint
        else:
            return v


def _reweight(
    samples: Array, unbiased_joint: ParameterJoint, joint: ParameterJoint
) -> Array:
    batch_dim = samples[0].shape[0]
    logit = tf.zeros((batch_dim,))
    for i, (uj, j) in enumerate(zip(unbiased_joint, joint)):
        # reduce across other axis (summing independent variable log ps)
        logitdiff = uj.log_prob(samples[i] + EPS) - j.log_prob(samples[i] + EPS)
        logit += tf.reduce_sum(tf.reshape(logitdiff, (batch_dim, -1)), axis=1)
    return tf.math.softmax(logit)


class TrainableInputLayer(tf.keras.layers.Layer):
    """Create trainable input layer for :py:class:`tfp.distributions.Distribution`

    This will, given a fake input, return a trainable weight set by ``initial_value``. Use
    to feed into distributions that can be trained.

    :param initial_value: starting value, determines shape/dtype of output
    :param constraint: Callable that returns scalar given output. See :py:class:`tf.keras.layers.Layer`
    :param kwargs: See :py:class:`tf.Keras.layers.Layer` for additional arguments
    """

    def __init__(
        self,
        initial_value: Array,
        constraint: Callable[[Array], float] = None,
        **kwargs
    ):
        super(TrainableInputLayer, self).__init__(**kwargs)
        flat = initial_value.flatten()
        self.initial_value = initial_value
        self.w = self.add_weight(
            "value",
            shape=initial_value.shape,
            initializer=tf.constant_initializer(flat),
            constraint=constraint,
            dtype=self.dtype,
            trainable=True,
        )

    def call(self, inputs: Array) -> Array:
        """See call of :class:`tf.keras.layers.Layer`"""
        batch_dim = tf.shape(inputs)[:1]
        return tf.tile(
            self.w[tf.newaxis, ...],
            tf.concat((batch_dim, tf.ones(tf.rank(self.w), dtype=tf.int32)), axis=0),
        )


class HyperMaxentModel(MaxentModel):
    """Keras Maximum entropy model

    :param restraints: List of :class:`Restraint`
    :param prior_model: :class:`ParameterJoint` that specifies prior
    :param simulation: Callable that will generate observations given the output from ``prior_model``
    :param reweight: True means use to remove effect of prior training updates via reweighting, which keeps as close as possible to given untrained ``prior_model``
    :param name: Name of model
    """

    def __init__(
        self,
        restraints: List[Restraint],
        prior_model: ParameterJoint,
        simulation: Callable[[Array], Array],
        reweight: bool = True,
        name: str = "hyper-maxent-model",
        **kwargs
    ):
        super(HyperMaxentModel, self).__init__(
            restraints=restraints, name=name, **kwargs
        )
        self.prior_model = prior_model
        self.reweight = reweight
        self.unbiased_joint = prior_model(tf.constant([1.0]))
        # self.trajs = trajs
        if hasattr(self.unbiased_joint, "sample"):
            self.unbiased_joint = [self.unbiased_joint]
        self.simulation = simulation

    def fit(
        self,
        sample_batch_size: int = 256,
        final_batch_multiplier: int = 4,
        param_epochs: int = None,
        outer_epochs: int = 10,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """Fit to given outcomes from ``simulation``

        :param sample_batch_size: Number of observations to sample per ``outer_epochs``
        :param final_batch_multiplier: Sets number of final MaxEnt fitting step after training ``prior_model``. Final number of MaxEnt steps will be ``final_batch_multiplier * sample_batch_size``
        :param param_epochs: Number of times ``prior_model`` will be fit to sampled observations
        :param outer_epochs: Number of loops of sampling/``prior_model`` fitting
        :param kwargs: See :class:tf.keras.Model ``fit`` method for further optional arguments, like ``verbose=0`` to hide output
        :return: The :class:`tf.keras.callbacks.History` of fit
        """
        me_history, prior_history = None, None

        # backwards compatible for my bad spelling
        if "outter_epochs" in kwargs:
            outer_epochs = kwargs["outter_epochs"]
            del kwargs["outter_epochs"]

        # we want to reset optimizer state each time we have
        # new trajectories
        # but compile, new object assignment both
        # don't work.
        # So I guess use SGD?

        def new_optimizer():
            return self.optimizer.__class__(**self.optimizer.get_config())

        uni_flags = ["verbose"]
        prior_kwargs = {}
        for u in uni_flags:
            if u in kwargs:
                prior_kwargs[u] = kwargs[u]

        if param_epochs is None:
            param_epochs = 10
            if "epochs" in kwargs:
                param_epochs = kwargs["epochs"]
        for i in range(outer_epochs - 1):
            # sample parameters
            psample, y, joint = self.prior_model.sample(sample_batch_size, True)
            trajs = self.simulation(*psample)
            try:
                if trajs.shape[0] != sample_batch_size:
                    raise ValueError(
                        "Simulation must take in batched samples and return batched outputs"
                    )
            except TypeError as e:
                raise ValueError(
                    "Simulation must take in batched samples and return batched outputs"
                )
            # get reweight, so we keep original parameter
            # probs
            rw = _reweight(y, self.unbiased_joint, joint)
            # TODO reset optimizer state
            if self.reweight:
                hm = super(HyperMaxentModel, self).fit(trajs, rw, **kwargs)
            else:
                hm = super(HyperMaxentModel, self).fit(trajs, **kwargs)
            fake_x = tf.constant(sample_batch_size * [1.0])
            hp = self.prior_model.fit(
                fake_x,
                y,
                sample_weight=self.traj_weights,
                epochs=param_epochs,
                **prior_kwargs
            )
            if me_history is None:
                me_history = hm
                prior_history = hp
            else:
                me_history = _merge_history(me_history, hm)
                prior_history = _merge_history(prior_history, hp)

        # For final fit use more samples
        outs = []
        rws = []
        for i in range(final_batch_multiplier):
            psample, y, joint = self.prior_model.sample(sample_batch_size, True)
            trajs = self.simulation(*psample)
            outs.append(trajs)
            rw = _reweight(y, self.unbiased_joint, joint)
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
        me_history = _merge_history(me_history, hm)
        return _merge_history(me_history, prior_history, "prior-")
