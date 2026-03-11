"""
Command-line interface to some functions in :py:mod:`~labrador.utils`.

Entry points are defined in pyproject.toml under [project.scripts].
"""
import argparse
from pathlib import Path
from .. import utils


def setup_rundir_and_priordir_cli():
    """
    CLI to :py:func:`~labrador.utils.setup_rundir` follwed by
    :py:func:`~labrador.utils.setup_priordir`.

    This function is available as ``lab-setup-rundir-and-priordir``.
    """
    parser = argparse.ArgumentParser(
        description='Create a new run directory under the given parent.')
    parser.add_argument('parentdir', type=Path, help='Parent directory')
    parser.add_argument('--prefix', type=str, default='run_',
                        help='Prefix for the run directory (default: run_)')

    args = parser.parse_args()

    rundir = utils.setup_rundir(**vars(args))
    priordir = utils.setup_priordir(rundir)

    print(f'rundir: {rundir}')
    print(f'priordir: {priordir}')


def setup_priordir_cli():
    """
    CLI to :py:func:`~labrador.utils.setup_priordir`.

    This function is available as ``lab-setup-priordir``.
    """
    parser = argparse.ArgumentParser(
        description='Create a new prior directory under the given rundir.')
    parser.add_argument('rundir', type=Path, help='Run directory')
    parser.add_argument(
        '--prefix', type=str, default='prior_',
        help='Prefix for the prior directory (default: prior_)')

    args = parser.parse_args()
    priordir = utils.setup_priordir(**vars(args))
    print(f'priordir: {priordir}')


def setup_rescalerdir_cli():
    """
    CLI to :py:func:`~labrador.utils.setup_rescalerdir`.

    This function is available as ``lab-setup-rescalerdir``.
    """
    parser = argparse.ArgumentParser(
        description='Create a new rescaler directory under a given priordir.')
    parser.add_argument('priordir', type=Path, help='Physical-prior directory')
    parser.add_argument(
        '--prefix', type=str, default='rescaler_',
        help='Prefix for the rescaler directory (default: rescaler_)')

    args = parser.parse_args()
    rescalerdir = utils.setup_rescalerdir(**vars(args))
    print(f'rescalerdir: {rescalerdir}')


def setup_sbidir_cli():
    """
    CLI to :py:func:`~labrador.utils.setup_sbidir`.

    This function is available as ``lab-setup-sbidir``.
    """
    parser = argparse.ArgumentParser(
        description='Create a new SBI directory under a given rescalerdir.')
    parser.add_argument('rescalerdir', type=Path, help='Rescaler directory')
    parser.add_argument(
        '--prefix', type=str, default='sbi_',
        help='Prefix for the sbi directory (default: sbi_)')

    args = parser.parse_args()
    sbidir = utils.setup_sbidir(**vars(args))
    print(f'sbidir: {sbidir}')


def setup_unfolderdir_cli():
    """
    CLI to :py:func:`~labrador.utils.setup_unfolderdir`.

    This function is available as ``lab-setup-unfolderdir``.
    """
    parser = argparse.ArgumentParser(
        description='Create a new unfolder directory in a given rescalerdir.')
    parser.add_argument('rescalerdir', type=Path, help='Rescaler directory')
    parser.add_argument(
        '--prefix', type=str, default='unfolder_',
        help='Prefix for the unfolder directory (default: unfolder_)')

    args = parser.parse_args()
    unfolderdir = utils.setup_unfolderdir(**vars(args))
    print(f'unfolderdir: {unfolderdir}')


def save_model():
    """
    CLI to :py:func:`~labrador.utils.Tree.to_tar`.

    This function is available as ``lab-save-model``.
    """
    parser = argparse.ArgumentParser(
    description='Export a trained model to tar.gz.')
    parser.add_argument('sbidir', type=Path, help='SBI directory')
    parser.add_argument('unfolderdir', type=Path, help='Unfolder directory')
    parser.add_argument(
        '--filename', type=Path, default=None,
        help='Output path. By default it is created alongside `rundir`.')

    args = parser.parse_args()

    tree = utils.Tree(args.sbidir, args.unfolderdir)
    tree.to_tar(args.filename or tree.rundir.parent)
