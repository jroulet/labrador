"""
Command-line interface to some functions in :py:mod:`~cogwheel_machine.utils`.
"""
import argparse
from pathlib import Path
from cogwheel_machine import utils


def setup_rundir_cli():
    """CLI to :py:func:`~cogwheel_machine.utils.setup_rundir`."""
    parser = argparse.ArgumentParser(
        description='Create a new run directory under the given parent.')
    parser.add_argument('parentdir', type=Path, help='Parent directory')
    parser.add_argument('--prefix', type=str, default='run_',
                        help='Prefix for the run directory (default: run_)')

    args = parser.parse_args()
    rundir = utils.setup_rundir(**vars(args))
    print(f'rundir: {rundir}')
