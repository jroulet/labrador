"""Command line interface to ``simulation.merge_chunks``."""
import argparse
from cogwheel_machine import simulation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge chunks of data.")
    parser.add_argument("rundir", type=str, help="Run directory")
    parser.add_argument("--delete_chunks_after_merging", action="store_true",
                        help="Delete chunks after merging")

    simulation.merge_chunks(**vars(parser.parse_args()))
