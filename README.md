# Maximum Entropy Inference


![tests](https://github.com/ur-whitelab/maxent/actions/workflows/test.yml/badge.svg) ![paper](https://github.com/ur-whitelab/maxent/actions/workflows/paper.yml/badge.svg) [![docs](https://github.com/ur-whitelab/maxent/actions/workflows/docs.yml/badge.svg)](https://ur-whitelab.github.io/maxent/)

This provides a Keras implementation of maximum entropy simulation based inference. The point of this package is to reweight outcomes from a simulator to agree with observations, rather than trying to optimize your simulators input parameters. The simulator must necessarily give multiple outcomes - either because you're trying multiple sets of input parameters or it has intrinsic noise. The assumption of this model is that your simulator is approximately correct. The observations being fit could have come the distribution of outcomes of your simulator.

## About maximum entropy

Maximum entropy reweighting is a straightforward black box method that can be applied to arbitrary simulators with few observations. Its runtime is independent of the number of parameters used by the simulator, and it has been shown analytically to minimally change the prior to agree with observations. This method fills a niche in the small-data, high-complexity regime of SBI parameter inference, because it accurately and minimally biases a prior to match observations and does not scale in runtime with the number of model parameters.

## Installation

```sh
pip install maxent-infer
```

## Quick Start

### A Pandas Data Frame

Consider a data frame representing outcomes from our prior model/simulator. We would like to
regress these outcomes to data.

```python
import pandas as pd
import numpy as np
import maxent


data = pd.read_csv('data.csv')
```

Perhaps we have a single observation we would like to match. We can define it with a restraint. Let's say
the observation corresponds to the values in column 3.

```python

def observe(single_row):
  return single_row[3]

r = maxent.Restraint(observe, target=1.5)
```

Now we'll fit using MaxEnt
```python
model = maxent.MaxentModel(r)
model.compile()
model.fit(data.values)
```

We now have a set of weights -- one per row -- that we can use to compute other expressions.
For example, here is the most likely outcome (mode)

```python
i = np.argmax(model.traj_weights)
mode = data.iloc[i, :]
```

Here are the new column averages
```python
col_avg = np.sum(data.values * model.traj_weights[:, np.newaxis], axis=0)
```

### A simulator

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

![image](https://user-images.githubusercontent.com/908389/130389256-2710cb73-617f-4e71-b3ba-e32bd0f85d6a.png)


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

# plot showing weights of trajectories:
```

![image](https://user-images.githubusercontent.com/908389/130389259-3a081e19-110a-4c80-9f91-3b3902444e21.png)


## Further Examples

You can find the examples used in the manuscript, including comparisons with competing methods: [here](https://ur-whitelab.github.io/maxent/toc.html). These examples use the latest package versions, so the figures will not exactly match those in the manuscript. If you would like to reproduce the manuscript exactly, install the packages in `paper/requirements.txt` and execute the notebooks in `paper` (this is the output from the `paper` workflow above).

## Citation

[See preprint](https://arxiv.org/abs/2104.09668) and the citation:

```bibtex
@article{barrett2022simulation,
  title={Simulation-Based Inference with Approximately Correct Parameters via Maximum Entropy},
  author={Barrett, Rainier and Ansari, Mehrad and Ghoshal, Gourab and White, Andrew D},
  journal={Machine Learning: Science and Technology},
  year={2022}
}
```

## License

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
