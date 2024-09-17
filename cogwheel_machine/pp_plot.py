"""P-P plots."""
import multiprocessing
from pathlib import Path
from scipy import stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from cogwheel import gw_plotting

from cogwheel_machine import utils


CREDIBLE_INTERVALS_FILENAME = 'credible_intervals.feather'

LATEX_LABELS = gw_plotting.plotting.LatexLabels(
    gw_plotting.CornerPlot.DEFAULT_LATEX_LABELS | {
        'diff_regularized0pn': r'$\Delta_{\rm 0PN}$',
        'dt_refdet': r'$\Delta t_{\rm ref\,det}$',
        'relative_dhat': r'$A_{\rm ref\,det}\hat d$'})


def pp_plot(credible_intervals, ax=None, show_legend=True,
            show_sigmas=(1, 2)):
    """
    Draw a P-P plot.

    Parameters
    ----------
    credible_intervals: pandas.DataFrame
        E.g. the output of ``get_credible_intervals``.

    ax: matplotlib.axes.Axes, optional
        Where to draw the P-P plot

    show_legend: bool
        Whether to display the legend.

    show_sigmas: sequence of float
        Add shaded regions to the plot with the corresponding
        expected uncertainty.
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot((0, 1), (0, 1), 'k:')  # Reference diagonal line

    # P-P traces:
    for par in credible_intervals:
        sorted_credible_intervals = np.sort(credible_intervals[par])
        ax.plot(sorted_credible_intervals,
                np.linspace(0, 1, len(credible_intervals)),
                label=LATEX_LABELS[par], lw=1.2)


    if show_legend:
        ax.legend(fontsize=10, frameon=True, framealpha=.5, labelspacing=0.25,
                  loc='upper left', edgecolor='none', borderpad=0.3)

    for sigmas in show_sigmas:
        plt.fill_between(*_pp_error(sigmas, len(credible_intervals)),
                         color='k', lw=1, ls=':', zorder=0, alpha=.1)

    ax.set_title(f'$N = {len(credible_intervals)}$', fontsize='medium')
    ax.set_xlabel('Credible interval')
    ax.set_ylabel('Fraction of injections in credible interval')

    ax.tick_params(axis='x', direction='in', top=True)
    ax.tick_params(axis='y', direction='in', right=True)
    ax.grid(linestyle=':')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def get_credible_intervals(modeldir, load=True, save=True, n_data=None,
                           n_samples=1000, n_processes=20):
    """
    Load or compute credible intervals, needed to construct a P-P plot.

    Parameters
    ----------
    modeldir: os.PathLike
        Directory containing a trained posterior.

    load: bool
        Whether to load credible intervals from disk if available.

    save: bool
        Whether to save credible intervals to disk in case they had to
        be computed.

    n_data: int
        How many injections to do or load.

    n_samples: int
        How many samples to use for computing each credible interval, in
        case we have to compute them.

    n_processes: int
        How many processes to use in case we have to compute the
        credible intervals.

    Return
    ------
    pd.DataFrame
        Columns contain parameters, rows contain floats between 0 and 1
        representing the credible interval at which the truth is
        recovered in the posterior.

    See Also
    --------
    pp_plot
    """
    modeldir = Path(modeldir)

    filepath = modeldir/CREDIBLE_INTERVALS_FILENAME
    if load and filepath.exists():
        return pd.read_feather(filepath)[:n_data]

    credible_intervals = _compute_credible_intervals(
        modeldir, n_data, n_samples, n_processes)

    if save:
        credible_intervals.to_feather(filepath)

    return credible_intervals


def _compute_credible_intervals(modeldir, n_data, n_samples, n_processes
                               ) -> pd.DataFrame:
    folded_sampled_params, data = _load_data(modeldir.parent, n_data)

    posterior = torch.load(modeldir/'posterior.pt',
                           map_location=torch.device('cpu'),
                           weights_only=False)

    with multiprocessing.Pool(n_processes) as pool:
        injections = (row for _, row in folded_sampled_params.iterrows())

        credible_intervals = pool.starmap(
            _compute_credible_interval,
            ((injection, x_obs, posterior, n_samples)
             for injection, x_obs in zip(injections, data)))

    return pd.DataFrame.from_records(credible_intervals)


def _compute_credible_interval(injection, x_obs, posterior, n_samples
                              ) -> dict:
    """
    Return credible interval for various parameters corresponding to a
    single injection.

    Parameters
    ----------
    injection: dict-like
        Keys correspond to the physical parameters of the posterior.

    x_obs: float array
        Data.

    posterior: sbi.inference.posteriors.direct_posterior.DirectPosterior
        Posterior density estimator.

    n_samples: int
        How many samples to generate for computing the credible
        interval.
    """
    samples = pd.DataFrame(
        posterior.sample((n_samples,), x=x_obs, show_progress_bars=False),
        columns=injection.keys())

    return {par: np.mean(samples[par] < truth)
            for par, truth in injection.items()}


def _load_data(rundir, n_data):
    datadir = rundir/utils.TEST_DIR

    mask = np.load(datadir/utils.MASK_FILENAME)
    data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask][:n_data]

    rescaled_params = pd.DataFrame(
        np.load(datadir/utils.RESCALED_PARAMETERS_FILENAME)[:n_data],
        columns=_get_folded_params(rundir))

    return rescaled_params, data


def _get_folded_params(rundir):
    config = utils.load_data_config(rundir)

    params = list(config.TRANSFORM_CLASS.sampled_params)
    for par in config.TRANSFORM_CLASS.folded_params:
        params[params.index(par)] = f'folded_{par}'
    return params


def _pp_error(sigmas: float, n_sim: int):
    """Return ``x, y1, y2`` inputs to ``plt.fill_between``."""
    x_values = np.linspace(0, 1, 1000)
    cdfs = stats.norm.cdf((-sigmas, sigmas))
    y_values = stats.binom.ppf(cdfs[:, np.newaxis], n_sim, x_values) / n_sim
    return x_values, *y_values
