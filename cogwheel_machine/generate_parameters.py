"""Generate physical parameters for the training and test datasets."""
import argparse
import os
from pathlib import Path

import cogwheel.utils

from . import utils


def submit_condor(rundir,
                  request_cpus=1,
                  request_memory='1G',
                  request_disk='1G',
                  **submit_kwargs):
    """
    Submit an HTCondor job to generate simulation parameters.

    This will generate the following files:
        {submission_scripts}/generate_parameters.{sub,sh,out,err,log}

    Parameters
    ----------
    rundir: str, os.PathLike
        Simulations directory, should contain a file `data_config.py`

    request_cpus, request_memory, request_disk: int or str
        Specifications in the HTCondor submit file.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do
        not pass `executable`, `output`, `error`, `log`, `args`,
        `queue`, which will be dealt with automatically.
    """
    rundir = Path(rundir).resolve()
    _check_rundir(rundir)
    scripts_dir = rundir/'submission_scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    submit_kwargs = {
        'submit_path': scripts_dir/'generate_parameters.sub',
        'executable': scripts_dir/'generate_parameters.sh',
        'output': scripts_dir/'generate_parameters.out',
        'error': scripts_dir/'generate_parameters.err',
        'log': scripts_dir/'generate_parameters.log',
        'args': rundir.as_posix(),
        'request_cpus': request_cpus,
        'request_memory': request_memory,
        'request_disk': request_disk,
        } | submit_kwargs

    cogwheel.utils.submit_condor(**submit_kwargs)


def main(rundir):
    """
    Parameters
    ----------
    rundir: PathLike
        Path to a directory, should contain a file `data_config.py` with
        analysis choices.
        See ``cogwheel_machine/example_configs/data_config.py`` for
        an example.

    See also
    --------
    utils.setup_rundir
    """
    rundir = Path(rundir)
    _check_rundir(rundir)

    config = utils.load_data_config(rundir)

    prior = config.PRIOR_CLASS(**config.PRIOR_KWARGS)

    for datadir, n_simulations in [
            (rundir/utils.TRAINING_DIR, config.N_TRAINING_SIMULATIONS),
            (rundir/utils.TEST_DIR, config.N_TEST_SIMULATIONS)]:
        os.makedirs(datadir)
        simulation_parameters = prior.generate_random_samples(n_simulations)
        simulation_parameters.to_feather(datadir/utils.PARAMETERS_FILENAME)


def _check_rundir(rundir):
    for dirname in utils.TRAINING_DIR, utils.TEST_DIR:
        parameters_file = rundir/dirname/utils.PARAMETERS_FILENAME
        if parameters_file.exists():
            raise FileExistsError(f'{parameters_file} already exists!')

    utils.write_version(rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate injection parameters from a "training" prior.')
    parser.add_argument(
        'rundir', help='path to a directory containing a file `config.py`.')

    main(**vars(parser.parse_args()))
