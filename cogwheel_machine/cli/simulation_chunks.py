"""
Command-line interfaces for simulating the training set in chunks.

See entry points in the pyproject.toml file under [project.scripts].
"""
import argparse
from cogwheel_machine import simulation


def simulate_chunk_cli():
    """CLI to :py:func:`~cogwheel_machine.simulation.simulate_chunk`."""
    parser = argparse.ArgumentParser(description='Simulate a chunk of data.')
    parser.add_argument('datadir', type=str, help='Train or test directory')
    parser.add_argument('i_start', type=int, help='Start index for simulation')
    parser.add_argument('i_end', type=int, help='End index for simulation')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes to use')
    args = parser.parse_args()
    simulation.simulate_chunk(**vars(args))


def merge_chunks_cli():
    """CLI to :py:func:`~cogwheel_machine.simulation.merge_chunks`."""
    parser = argparse.ArgumentParser(description='Merge chunks of data.')
    parser.add_argument('rundir', type=str, help='Run directory')
    parser.add_argument('--delete_chunks_after_merging', action='store_true',
                        help='Delete chunks after merging')

    simulation.merge_chunks(**vars(parser.parse_args()))
