"""Command line interface to ``simulation.simulate_chunk``."""
import argparse
from cogwheel_machine import simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate a chunk of data.")
    parser.add_argument("datadir", type=str, help="Train or test directory")
    parser.add_argument("i_start", type=int, help="Start index for simulation")
    parser.add_argument("i_end", type=int, help="End index for simulation")
    parser.add_argument("--processes", type=int, default=1,
                        help="Number of processes to use")

    simulation.simulate_chunk(**vars(parser.parse_args()))
