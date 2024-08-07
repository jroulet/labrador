"""
Integration test of the modules for generating training data, namely:
	* generate_parameters
	* simulation
	* compression
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import tempfile
from unittest import TestCase, main

from cogwheel_machine import (compression,
							  generate_parameters,
							  simulation,
							  utils)


class TrainingDataTestCase(TestCase):
	"""Class to test the generation of training data."""
	def test_make_training_data(self):
		"""
		Generate a small amount of training data in a temporary
		directory.
		"""
		with tempfile.TemporaryDirectory() as datadir:
			sim_dir = utils.setup_sim_dir(datadir)
			generate_parameters.main(sim_dir)
			simulation.main(sim_dir)
			compression.create_mask(sim_dir)
			compression.svd_compression(sim_dir)


if __name__ == '__main__':
	main()
