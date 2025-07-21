"""P-P plots."""
import argparse
import multiprocessing
import warnings
from pathlib import Path
from scipy import stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from cogwheel import gw_plotting

from . import utils

warnings.filterwarnings('ignore',
                        message='torch.triangular_solve is deprecated')

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
    credible_intervals : pandas.DataFrame
        E.g. the output of ``get_credible_intervals``.
        Columns are parameter names, rows are injections. Values are the
        credible interval that our inference assigns to the truth.
        May contain an additional column 'weights' with the ratio of the
        physical prior to the simulation prior with which the credible
        intervals were obtained (up to an arbitrary normalization).

    ax : matplotlib.axes.Axes, optional
        Where to draw the P-P plot

    show_legend : bool
        Whether to display the legend.

    show_sigmas : sequence of float
        Add shaded regions to the plot with the corresponding
        expected uncertainty.
    """
    if ax is None:
        _, ax = plt.subplots()

    n_injections = len(credible_intervals)
    weights = credible_intervals.get('weights', np.ones(n_injections))
    weights = weights / weights.sum()

    credible_intervals = credible_intervals.drop(columns='weights',
                                                 errors='ignore')

    ax.plot((0, 1), (0, 1), 'k:')  # Reference diagonal line

    # P-P traces:
    for par, intervals in credible_intervals.items():
        order = intervals.argsort()
        sorted_credible_intervals = intervals.iloc[order]
        empirical_credible_intervals = weights[order].cumsum()
        ax.plot(sorted_credible_intervals,
                empirical_credible_intervals,
                label=LATEX_LABELS[par], lw=1.2)

    if show_legend:
        ax.legend(fontsize=10, frameon=True, framealpha=.5, labelspacing=0.25,
                  loc='upper left', edgecolor='none', borderpad=0.3)

    for sigmas in show_sigmas:
        plt.fill_between(*_pp_error(sigmas, n_injections),
                         color='k', lw=1, ls=':', zorder=0, alpha=.1)

    ax.set_title(f'$N = {n_injections}$', fontsize='medium')
    ax.set_xlabel('Credible interval')
    ax.set_ylabel('Fraction of injections in credible interval')

    ax.tick_params(axis='x', direction='in', top=True)
    ax.tick_params(axis='y', direction='in', right=True)
    ax.grid(linestyle=':')
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def get_credible_intervals(sbidir, load=True, save=True, n_data=None,
                           n_samples=1000, n_processes=20):
    """
    Load or compute credible intervals, needed to construct a P-P plot.

    Parameters
    ----------
    sbidir : os.PathLike
        Directory containing a trained posterior.

    load : bool
        Whether to load credible intervals from disk if available.

    save : bool
        Whether to save credible intervals to disk in case they had to
        be computed.

    n_data : int
        How many injections to do or load.

    n_samples : int
        How many samples to use for computing each credible interval, in
        case we have to compute them.

    n_processes : int
        How many processes to use in case we have to compute the
        credible intervals.

    Returns
    -------
    pd.DataFrame
        Columns contain parameters, rows contain floats between 0 and 1
        representing the credible interval at which the truth is
        recovered in the posterior.

    See Also
    --------
    pp_plot
    """
    sbidir = Path(sbidir)

    filepath = sbidir/CREDIBLE_INTERVALS_FILENAME
    if load and filepath.exists():
        return pd.read_feather(filepath)[:n_data]

    credible_intervals = _compute_credible_intervals(
        sbidir, n_data, n_samples, n_processes)

    if save:
        credible_intervals.to_feather(filepath)

    return credible_intervals


def _compute_credible_intervals(sbidir, n_data, n_samples, n_processes
                               ) -> pd.DataFrame:
    rescalerdir = sbidir.resolve().parent
    folded_sampled_params, data, weights = _load_data(rescalerdir, n_data)

    posterior = torch.load(sbidir/utils.POSTERIOR_FILENAME,
                           map_location=torch.device('cpu'),
                           weights_only=False)

    injections = (row for _, row in folded_sampled_params.iterrows())

    with multiprocessing.Pool(n_processes) as pool:
        credible_intervals = pool.starmap(
            _compute_credible_interval,
            ((injection, x_obs, posterior, n_samples)
             for injection, x_obs in zip(injections, data)))

    credible_intervals = pd.DataFrame.from_records(credible_intervals)
    credible_intervals['weights'] = weights
    return credible_intervals


def _compute_credible_interval(injection, x_obs, posterior, n_samples
                              ) -> dict:
    """
    Return credible interval for various parameters corresponding to a
    single injection.

    Parameters
    ----------
    injection : dict-like
        Keys correspond to the physical parameters of the posterior.

    x_obs : float array
        Data.

    posterior : sbi.inference.posteriors.direct_posterior.DirectPosterior
        Posterior density estimator.

    n_samples : int
        How many samples to generate for computing the credible
        interval.
    """
    samples = pd.DataFrame(
        posterior.sample((n_samples,), x=x_obs, show_progress_bars=False),
        columns=injection.keys())

    return {par: np.mean(samples[par] < truth)
            for par, truth in injection.items()}


def _load_data(rescalerdir, n_data):
    priordir, rundir = rescalerdir.parents[:2]
    datadir = rundir/utils.TEST_DIR
    rescaled_datadir = rescalerdir/utils.TEST_DIR

    mask = np.load(datadir/utils.MASK_FILENAME)
    data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask][:n_data]

    rescaled_params = pd.DataFrame(
        np.load(rescaled_datadir/utils.RESCALED_PARAMETERS_FILENAME)[:n_data],
        columns=_get_folded_params(rundir))

    weights = np.load(priordir/utils.TEST_DIR/utils.WEIGHTS_FILENAME)[:n_data]

    return rescaled_params, data, weights


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


def main(sbidir, load=True, save=True, n_data=None, n_samples=1000,
         n_processes=20):
    """Make a P-P plot and save it in `sbidir`."""
    sbidir = Path(sbidir).resolve()
    credible_intervals = get_credible_intervals(
        sbidir, load, save, n_data, n_samples, n_processes)

    pp_plot(credible_intervals)
    plt.title(f'Folded & rescaled; {sbidir.name}')
    plt.savefig(sbidir/'pp_plot.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a P-P plot.')
    parser.add_argument(
        'sbidir',
        type=str,
        help='Path to the model directory.'
    )
    parser.add_argument(
        '--n_data',
        type=int,
        default=2000,
        help='Number of simulations to use (default: 2000).'
    )
    parser.add_argument(
        '--n_processes',
        type=int,
        default=20,
        help='Number of processes to use (default: 20).'
    )

    main(**vars(parser.parse_args()))
