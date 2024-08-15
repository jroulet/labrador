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
		with tempfile.TemporaryDirectory() as parentdir:
			rundir = utils.setup_rundir(parentdir)
			generate_parameters.main(rundir)
			simulation.main(rundir)
			compression.create_mask(rundir)
			compression.svd_compression(rundir)
			print('Created these training data:')
			os.system(f'tree {parentdir}')


if __name__ == '__main__':
	main()
