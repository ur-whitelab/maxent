# Maximum Entropy Simulation Based Inference


![tests](https://github.com/ur-whitelab/maxent/actions/workflows/test.yml/badge.svg) ![paper](https://github.com/ur-whitelab/maxent/actions/workflows/paper.yml/badge.svg) [![docs](https://github.com/ur-whitelab/maxent/actions/workflows/docs.yml/badge.svg)](https://ur-whitelab.github.io/exmol/)

This provides a Keras implementation of maximum entropy simulation based inference. The point of this package is to reweight outcomes from a simulator to agree with observations, rather than trying to optimize your simulators input parameters. The simulator must necessarily give multiple outcomes - either because you're trying multiple sets of input parameters or it has intrinsic noise. The assumption of this model is that your simulator is approximately correct. The observations being fit could have come the distribution of outcomes of your simulator.

## About maximum entropy

Maximum entropy reweighting is a straightforward black box method that can be applied to arbitrary simulators with few observations. Its runtime is independent of the number of parameters used by the simulator, and it has been shown analytically to minimally change the prior to agree with observations. This method fills a niche in the small-data, high-complexity regime of SBI parameter inference, because it accurately and minimally biases a prior to match observations and does not scale in runtime with the number of model parameters.

## Installation

The package uses Keras (Tensorflow). To install:

```sh
pip install maxent@git+git://github.com/ur-whitelab/maxent.git
```

## Quick Start

```python
# assume z contains outcomes from your simulator of arbitrary shape, with first axis being batch
z = np.random.normal(size=256).astype(np.float32)

# restrain these observations so that second moment average is 2
r = maxent.Restraint(fxn=lambda x: x**2, target=2, prior=maxent.EmptyPrior())

# create model by passing in restraints as list
model = maxent.MaxentModel([r])

# usual Keras fitting
model.compile(tf.keras.optimizers.Adam(0.1), 'mean_squared_error')
model.fit(z, epochs=128, verbose=0)

# now measure new mean of data, with reweighting
mean = np.sum(z * model.traj_weights)
```

## Citation

[Barrett, Rainier, et al. "Simulation-Based Inference with Approximately Correct Parameters via Maximum Entropy." arXiv preprint arXiv:2104.09668 (2021).](https://arxiv.org/abs/2104.09668)

```bibtex
@article{barrett2021simulation,
  title={Simulation-Based Inference with Approximately Correct Parameters via Maximum Entropy},
  author={Barrett, Rainier and Ansari, Mehrad and Ghoshal, Gourab and White, Andrew D},
  journal={arXiv preprint arXiv:2104.09668},
  year={2021}
}
```

## License

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
