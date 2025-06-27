"""Generate physical parameters for the training and test datasets."""
import argparse
import os
from pathlib import Path

import pandas as pd
import scipy.stats.qmc

from cogwheel.prior import FixedPrior, UniformPriorMixin, CombinedPrior

from . import condor_utils, utils


def _is_uniform_prior(cls):
    """
    Return whether `cls` corresponds to a uniform prior.

    That is: ``True`` if `cls` is a subclass of ``UniformPriorMixin``,
    ``FixedPrior``, or a ``CombinedPrior`` that combines only these;
    ``False`` otherwise.
    """
    # TODO move to cogwheel.prior
    if issubclass(cls, CombinedPrior):
        return all(_is_uniform_prior(prior_class)
                   for prior_class in cls.prior_classes)

    if issubclass(cls, UniformPriorMixin) or issubclass(cls, FixedPrior):
        return True

    return False


def _generate_qmc_samples(prior, n_samples, seed=None):
    """
    Sample the parameter space uniformly.

    Parameters
    ----------
    n_samples : int
        How many samples to generate.

    seed:
        Passed to ``numpy.default_rng``, for reproducibility.

    Returns
    -------
    pd.DataFrame with columns per
    ``.sampled_params + .standard_params``, with samples distributed
    uniformly.
    """
    # TODO move to cogwheel.prior
    if not _is_uniform_prior(prior.__class__):
        raise RuntimeError(f'{prior} is not a uniform prior!')

    samples = pd.DataFrame(
        prior.cubemin + prior.cubesize * scipy.stats.qmc.Halton(
            len(prior.sampled_params), seed=seed).random(n_samples),
        columns=prior.sampled_params)

    prior.transform_samples(samples)
    return samples


def setup_condor_sub(rundir, request_disk='8G', request_memory='8G',
                     submit=False, **submit_kwargs):
    """
    Create a script to run the generate_parameters job on HTCondor.

    This will generate the following files:
        {rundir}/submission_scripts/generate_parameters.{sub,sh}

    Parameters
    ----------
    rundir : os.PathLike
        Simulations directory, should contain a file `data_config.py`.

    request_disk : str
        Disk request for the HTCondor job. Default is '8G'.

    request_memory : str
        Memory request for the HTCondor job. Default is '8G'.

    submit : bool
        If True, submit the job to HTCondor. Otherwise just create the
        submission and executable files.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do
        not pass `executable`, `output`, `error`, `log`, `arguments`,
        `queue`, which will be dealt with automatically.

    Returns
    -------
    pathlib.Path
        Path to the HTCondor submit file.
    """
    rundir = Path(rundir).resolve()
    stem = rundir/'submission_scripts'/'generate_parameters'
    module = 'cogwheel_machine.generate_parameters'
    submit_path = condor_utils.setup_condor_sub(stem, module,
                                                request_memory=request_memory,
                                                request_disk=request_disk,
                                                submit=submit,
                                                arguments=rundir,
                                                **submit_kwargs)
    return submit_path


def main(rundir):
    """
    Parameters
    ----------
    rundir : PathLike
        Path to a directory, should contain a file `data_config.py` with
        analysis choices.
        See ``cogwheel_machine/example_configs/data_config.py`` for
        an example.

    See Also
    --------
    utils.setup_rundir
    """
    rundir = Path(rundir)
    _check_rundir(rundir)

    config = utils.load_data_config(rundir)

    prior = config.PRIOR_CLASS(**config.PRIOR_KWARGS)

    # Test set:
    _write_datadir(prior,
                   rundir/utils.TEST_DIR,
                   config.N_TEST_SIMULATIONS,
                   qmc=False)  # Two different quasirandom sequences can
                               # have weird correlations.

    # Training set:
    _write_datadir(prior,
                   rundir/utils.TRAINING_DIR,
                   config.N_TRAINING_SIMULATIONS,
                   qmc=config.QMC)


def _check_rundir(rundir):
    for dirname in utils.TRAINING_DIR, utils.TEST_DIR:
        parameters_file = rundir/dirname/utils.PARAMETERS_FILENAME
        if parameters_file.exists():
            raise FileExistsError(f'{parameters_file} already exists!')

    utils.write_version(rundir)


def _write_datadir(prior, datadir, n_simulations, qmc):
    if qmc:
        simulation_parameters = _generate_qmc_samples(prior, n_simulations)
    else:
        simulation_parameters = prior.generate_random_samples(n_simulations)

    os.makedirs(datadir, exist_ok=True)
    simulation_parameters.to_feather(datadir/utils.PARAMETERS_FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate injection parameters from a "training" prior.')
    parser.add_argument(
        'rundir', help='path to a directory with a file `data_config.py`.')

    main(**vars(parser.parse_args()))
