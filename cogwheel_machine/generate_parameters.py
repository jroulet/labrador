import argparse
from pathlib import Path

import cogwheel.utils

from . import utils


def submit_condor(sim_dir,
                  request_cpus=1,
                  request_memory='1G',
                  request_disk='1G',
                  **submit_kwargs):
    """
    Submit an HTCondor job to generate simulation parameters.

    This method generates 'generate_parameters.{sub,sh,out,err,log}',
    files, the user should provide any instructions for the submit file
    as `**submit_kwargs`.

    Parameters
    ----------
    sim_dir: str, os.PathLike
        Simulations directory, should contain a file `config.py`

    request_cpus, request_memory, request_disk: int or str
        Specifications in the HTCondor submit file.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do
        not pass `executable`, `output`, `error`, `log`, `args`,
        `queue`, which will be dealt with automatically.
    """
    sim_dir = Path(sim_dir).resolve()
    _check_sim_dir(sim_dir)

    submit_kwargs = {
        'submit_path': sim_dir/'generate_parameters.sub',
        'executable': sim_dir/'generate_parameters.sh',
        'output': sim_dir/'generate_parameters.out',
        'error': sim_dir/'generate_parameters.err',
        'log': sim_dir/'generate_parameters.log',
        'args': sim_dir.as_posix(),
        'request_cpus': request_cpus,
        'request_memory': request_memory,
        'request_disk': request_disk,
        } | submit_kwargs

    cogwheel.utils.submit_condor(**submit_kwargs)


def main(sim_dir):
    """
    Parameters
    ----------
    sim_dir: PathLike
        Path to a directory, should contain a file `config.py` with
        analysis choices. See ``cogwheel_machine/example_config.py`` for
        an example.
    """
    sim_dir = Path(sim_dir)
    _check_sim_dir(sim_dir)

    config = utils.load_config(sim_dir)

    prior = config.PRIOR_CLASS(**config.PRIOR_KWARGS)
    simulation_parameters = prior.generate_random_samples(config.N_SIMULATIONS)

    simulation_parameters.to_feather(sim_dir/utils.PARAMETERS_FILENAME)


def _check_sim_dir(sim_dir):
    parameters_file = sim_dir/utils.PARAMETERS_FILENAME
    if parameters_file.exists():
        raise FileExistsError(f'{parameters_file} already exists!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate injection parameters from a "training" prior.')
    parser.add_argument(
        'sim_dir', help='path to a directory containing a file `config.py`.')

    main(**vars(parser.parse_args()))
