import maxent
import unittest
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

np.random.seed(0)
tf.random.set_seed(0)


class TestPriors(unittest.TestCase):

    def test_empty(self):
        p = maxent.EmptyPrior()
        assert p.expected(1) == 0

    def test_laplace(self):
        p = maxent.Laplace(0.1)

    def test_restraint(self):
        r = maxent.Restraint(lambda x: x**2, 4, maxent.EmptyPrior())
        assert r(2) == 0


class TestLayers(unittest.TestCase):

    def test_rw_layer(self):
        l = maxent.ReweightLayer(10)
        w = l(np.arange(10, dtype=np.float32))
        assert len(w) == 1

    def test_avg_layer(self):
        l = maxent.ReweightLayer(10)
        la = maxent.AvgLayer(l)
        gk = np.arange(10, dtype=np.float32)
        w = l(gk)
        la(gk, w)

    def test_lrw_layer(self):
        l = maxent.ReweightLayerLaplace(
            np.random.normal(size=10).astype(np.float32))
        w = l(np.arange(10, dtype=np.float32))
        assert len(w) == 1

    def test_lavg_layer(self):
        l = maxent.ReweightLayerLaplace(
            np.random.normal(size=10).astype(np.float32))
        la = maxent.AvgLayerLaplace(l)
        gk = np.arange(10, dtype=np.float32)
        w = l(gk)
        la(gk, w)


class TestModel(unittest.TestCase):
    def test_me(self):
        data = np.random.normal(size=256).astype(np.float32)
        r = maxent.Restraint(lambda x: x**2, 2, maxent.EmptyPrior())
        model = maxent.MaxentModel([r])
        model.compile(tf.keras.optimizers.Adam(0.1), 'mean_squared_error')
        model.fit(data, epochs=128, verbose=0)
        # check we fit somewhat close
        e = np.sum(data**2 * model.traj_weights)
        npt.assert_array_almost_equal(e, 2.0, decimal=2)

    def test_lme(self):
        data = np.random.normal(size=256).astype(np.float32)
        r = maxent.Restraint(lambda x: x**2, 2, maxent.Laplace(0.01))
        model = maxent.MaxentModel([r])
        model.compile(tf.keras.optimizers.Adam(0.1), 'mean_squared_error')
        model.fit(data, epochs=128, verbose=0)
        # check we fit somewhat close
        e = np.sum(data**2 * model.traj_weights)
        npt.assert_array_almost_equal(e, 2.0, decimal=1)


class TestHyperModel(unittest.TestCase):
    def test_hme(self):
        # make a model for sampling parameters
        tf.random.set_seed(0)
        x = np.array([1., 1.])
        i = tf.keras.Input((1,))
        l = maxent.TrainableInputLayer(x)(i)
        d = tfp.layers.DistributionLambda(lambda x: tfd.Normal(
            loc=x[..., 0], scale=tf.math.exp(x[..., 1])))(l)
        model = maxent.ParameterJoint([lambda x: x], inputs=i, outputs=[d])
        model.compile(tf.keras.optimizers.SGD(0.1))

        # make simulator
        def simulate(x):
            y = np.random.normal(loc=x, scale=0.1)
            return y

        # make ME model
        r = maxent.Restraint(lambda x: x, 8, maxent.EmptyPrior())
        hme_model = maxent.HyperMaxentModel([r], model, simulate)
        hme_model.compile(tf.keras.optimizers.Adam(0.5), 'mean_squared_error')
        hme_model.fit(epochs=64, outter_epochs=2)
        e = np.sum(hme_model.trajs[:, 0] * hme_model.traj_weights)
        assert abs(e - 8.0) < 0.25


    def test_error(self):
        # make a model for sampling parameters
        x = np.array([1., 1.])
        i = tf.keras.Input((1,))
        l = maxent.TrainableInputLayer(x)(i)
        d = tfp.layers.DistributionLambda(lambda x: tfd.Normal(
            loc=x[..., 0], scale=tf.math.exp(x[..., 1])))(l)
        model = maxent.ParameterJoint([lambda x: x], inputs=i, outputs=[d])
        model.compile(tf.keras.optimizers.Adam(0.1))

        # make bad simulator
        def simulate(x):
            y = np.random.normal(loc=2, scale=0.1)
            return y

        # make ME model
        r = maxent.Restraint(lambda x: x, 8, maxent.EmptyPrior())
        hme_model = maxent.HyperMaxentModel([r], model, simulate)
        with self.assertRaises(ValueError) as e:
            hme_model.fit(epochs=1, outter_epochs=1)


if __name__ == '__main__':
    unittest.main()
