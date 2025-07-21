"""
Command-line interface to some functions in :py:mod:`~labrador.utils`.
"""
import argparse
from pathlib import Path
from . import utils


def setup_rundir_cli():
    """CLI to :py:func:`~labrador.utils.setup_rundir`."""
    parser = argparse.ArgumentParser(
        description='Create a new run directory under the given parent.')
    parser.add_argument('parentdir', type=Path, help='Parent directory')
    parser.add_argument('--prefix', type=str, default='run_',
                        help='Prefix for the run directory (default: run_)')

    args = parser.parse_args()
    rundir = utils.setup_rundir(**vars(args))
    print(f'rundir: {rundir}')


def setup_rescalerdir_cli():
    """CLI to :py:func:`~labrador.utils.setup_rescalerdir`."""
    parser = argparse.ArgumentParser(
        description='Create a new rescaler directory under a given priordir.')
    parser.add_argument('priordir', type=Path, help='Physical-prior directory')
    parser.add_argument(
        '--prefix', type=str, default='rescaler_',
        help='Prefix for the rescaler directory (default: rescaler_)')

    args = parser.parse_args()
    rescalerdir = utils.setup_rescalerdir(**vars(args))
    print(f'rescalerdir: {rescalerdir}')
