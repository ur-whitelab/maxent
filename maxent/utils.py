import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import core


class TransitionMatrix:
    def __init__(self, compartment_names, infectious_compartments):
        self.names = compartment_names
        self.infectious_compartments = infectious_compartments
        self.transitions = []
        self.mat = None

    def add_transition(self, name1, name2, time, time_var):
        if name1 not in self.names or name2 not in self.names:
            raise ValueError('name not in compartment names')
        if name1 == name2:
            raise ValueError('self-loops are added automatically')
        self.transitions.append([name1, name2, time, time_var])
        self.mat = None

    def prior_matrix(self):
        C = len(self.names)
        T1, T2 = np.zeros((C, C)), np.zeros((C, C))
        for n1, n2, v, vv in self.transitions:
            i = self.names.index(n1)
            j = self.names.index(n2)
            T1[i, j] = v
            T2[i, j] = vv
        return T1, T2

    def _make_matrix(self):

        C = len(self.names)
        T = np.zeros((C, C))
        for n1, n2, v, vv in self.transitions:
            i = self.names.index(n1)
            j = self.names.index(n2)
            T[i, j] = 1 / v
            # get what leaves
        np.fill_diagonal(T, 1 - np.sum(T, axis=1))
        self.mat = T

    @property
    def value(self):
        '''Return matrix value
        '''
        if self.mat is None:
            self._make_matrix()
        return self.mat


def _weighted_quantile(values, quantiles, sample_weight=None,
                       values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def patch_quantile(trajs, *args, figsize=(18, 18), patch_names=None, ** kw_args):
    '''does traj_quantile for trajectories of shape [ntrajs, time, patches, compartments]
    '''
    NP = trajs.shape[2]
    nrow = int(np.floor(np.sqrt(NP)))
    ncol = int(np.ceil(NP / nrow))
    print(f'Plotting {NP} patches in a {nrow} x {ncol} grid')
    fig, ax = plt.subplots(nrow, ncol, sharex=True,
                           sharey=True, figsize=figsize)
    for i in range(nrow):
        for j in range(ncol):
            if i * ncol + j == NP:
                break
            traj_quantile(trajs[:, :, i * ncol + j, :], *args, ax=ax[i, j],
                          add_legend=i == 0 and j == ncol - 1, **kw_args)
            ax[i, j].set_ylim(0, 1)
            if patch_names is None:
                ax[i, j].text(trajs.shape[1] // 2, 0.8,
                              f'Patch {i * ncol + j}')
            else:
                patch_names = patch_names
                ax[i, j].set_title(patch_names[i * ncol + j])

            if j == 0 and i == nrow // 2:
                ax[i, j].set_ylabel('Fraction')
            if i == nrow - 1 and j == ncol // 2:
                ax[i, j].set_xlabel('Time')
    plt.tight_layout()


def traj_quantile(trajs, weights=None, figsize=(9, 9), names=None, plot_means=True, ax=None, add_legend=True, add_title=None, alpha=0.6):
    '''Make a plot of all the trajectories and the average trajectory based on
      parameter weights.'''

    if names is None:
        names = [f'Compartment {i}' for i in range(trajs.shape[-1])]
    if weights is None:
        w = np.ones(trajs.shape[0])
    else:
        w = weights
    w /= np.sum(w)

    x = range(trajs.shape[1])

    # weighted quantiles doesn't support axis
    # fake it using apply_along
    qtrajs = np.apply_along_axis(lambda x: _weighted_quantile(
        x, [1/3, 1/2, 2/3], sample_weight=w), 0, trajs)
    if plot_means:
        # approximate quantiles as distance from median applied to mean
        # with clips
        mtrajs = np.sum(trajs * w[:, np.newaxis, np.newaxis], axis=0)
        qtrajs[0, :, :] = np.clip(
            qtrajs[0, :, :] - qtrajs[1, :, :] + mtrajs, 0, 1)
        qtrajs[2, :, :] = np.clip(
            qtrajs[2, :, :] - qtrajs[1, :, :] + mtrajs, 0, 1)
        qtrajs[1, :, :] = mtrajs
    if ax is None:
        ax = plt.gca()
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Fraction of Population')
    for i in range(trajs.shape[-1]):
        ax.plot(x, qtrajs[1, :, i],
                color=f'C{i}', label=f'Compartment {names[i]}')
        ax.fill_between(x, qtrajs[0, :, i], qtrajs[-1, :, i],
                        color=f'C{i}', alpha=alpha)
    if not plot_means:
        ax.plot(x, np.sum(qtrajs[1, :, :], axis=1),
                color='gray', label='Total', linestyle=':')

    if add_legend:
        # add margin for legend
        ax.set_xlim(0, max(x))
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


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


def compartment_restrainer(restrained_patches, restrained_compartments, npoints, ref_traj, prior, noise=0, start_time=None, end_time=None, time_average=7):
    number_of_restrained_compartments = len(restrained_compartments)
    number_of_restrained_patches = len(restrained_patches)
    M = ref_traj.shape[1]
    T = ref_traj.shape[0]
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = T//3
    print('Restraints are set on this time range: [{}, {}]'.format(
        start_time, end_time))
    # restrained_patches = np.random.choice(
    #     M, number_of_restrained_patches, replace=False)
    if number_of_restrained_patches > M:
        raise Exception(
            'Number of patches to be restrained exceeeds the total number of patches')
    restraints = []
    plot_fxns_list = []
    for i in range(number_of_restrained_patches):
        plot_fxns = []
        for j in range(number_of_restrained_compartments):
            res, plfxn = core.traj_to_restraints(ref_traj[start_time:end_time, :, :], [
                restrained_patches[i], restrained_compartments[j]], npoints, prior, noise, time_average)
            restraints += res
            plot_fxns += plfxn
        plot_fxns_list.append(plot_fxns)
    return restraints, plot_fxns_list
