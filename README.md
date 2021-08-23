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

Here we show how to take a random walk simulator and use `maxent` to have reweight the random walk so that the average end is at x = 2, y= 1.

```python
# simulate
def random_walk_simulator(T=10):
    x = [0,0]
    traj = np.empty((T,2))
    for i in range(T):
        traj[i] = x
        x += np.random.normal(size=2)
    return traj

N = 500
trajs = [random_walk_simulator() for _ in range(N)]

# now have N x T x 2 tensor
trajs = np.array(trajs)

# here is a plot of these trajectories
```

```python
# we want the random walk to have average end of 2,1
rx = maxent.Restraint(lambda traj: traj[-1,0], target=2)
ry = maxent.Restraint(lambda traj: traj[-1,1], target=1)

# create model by passing in restraints
model = maxent.MaxentModel([rx, ry])

# convert model to be differentiable/GPU (if available)
model.compile()
# fit to data
h = model.fit(trajs)

# can now compute other averages properties
# with new weights
model.traj_weights

# plot showing weight of trajectories:
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
