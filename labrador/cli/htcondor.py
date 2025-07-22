"""
Submit jobs to HTCondor for generating a training set.

1. Generate training and test parameters.
2. Simulate data in chunks.
3. Merge chunks into a single file.
4. Compress the merged file.

This script generates a DAGMan file to organize the simulation jobs and
submits it to HTCondor.
"""
import argparse
import textwrap
import subprocess
from pathlib import Path

from .. import compression, generate_parameters, simulation


def generate_data_cli():
    """
    Command-line interface for generating a training set.

    This function is accessible as ``lab-generate-data-htcondor``.
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''\
            Submit jobs to HTCondor for generating a training set.

            1. Generate training and test parameters.
            2. Simulate data in chunks.
            3. Merge chunks into single files.
            4. Compress the generated data.
            ''')
    )
    parser.add_argument('rundir', help='Run directory')
    parser.add_argument('--chunk-size', type=int, default=10_000,
                        help='Number of simulations performed by each job.')
    parser.add_argument(
        '--submit-arg', action='append', default=[],
        help='Extra submit file arguments as key=value pairs. '
             'Example: `--submit-arg accounting_group="my_acc_group"`.'
    )

    args = parser.parse_args()

    # Convert --submit-arg into a dict
    submit_kwargs = {}
    for pair in args.submit_arg:
        if '=' not in pair:
            raise ValueError(f'Invalid format for --submit-arg: {pair!r}')
        key, value = pair.split('=', 1)
        submit_kwargs[key] = value

    generate_data(rundir=args.rundir, chunk_size=args.chunk_size,
                  **submit_kwargs)


def generate_data(rundir, *, chunk_size=10_000, **submit_kwargs):
    """
    Submit jobs to HTCondor for generating a training set.

    1. Generate training and test parameters.
    2. Simulate data in chunks.
    3. Merge chunks into a single file.
    4. Compress the data.
    """
    # Generate submit files for all tasks
    submit_gen_parameters_path = generate_parameters.setup_condor_sub(
        rundir, **submit_kwargs)

    (submit_chunks_train_path, submit_chunks_test_path), submit_merge_path \
        = simulation.setup_condor_sub(
            rundir, chunk_size=chunk_size, **submit_kwargs)

    submit_compress_path = compression.setup_condor_sub(
        rundir, **submit_kwargs)

    # Generate DAGMan file for submitting jobs in the correct order
    dagman_path = _generate_dagman_file(submit_gen_parameters_path,
                                        submit_chunks_train_path,
                                        submit_chunks_test_path,
                                        submit_merge_path,
                                        submit_compress_path)

    # Submit DAGMan
    subprocess.run(['condor_submit_dag', dagman_path], check=True)
    print(f'Submitted DAGMan file: {dagman_path}')


def _generate_dagman_file(submit_gen_parameters_path,
                          submit_chunks_train_path,
                          submit_chunks_test_path,
                          submit_merge_path,
                          submit_compress_path
                          ) -> Path:
    """
    Create DAGMan file to organize simulation jobs and return its path.

    Parameters
    ----------
    submit_gen_parameters_path : os.PathLike
        Path to the submit file for generating parameters.

    submit_chunks_train_path : os.PathLike
        Path to the submit file for simulating training-set chunks.

    submit_chunks_test_path : os.PathLike
        Path to the submit file for simulating test-set chunks.

    submit_merge_path : os.PathLike
        Path to the submit file for merging chunks.

    submit_compress_path : os.PathLike
        Path to the submit file for compressing chunks.

    Returns
    -------
    Path
        Path to the generated DAGMan file.
    """
    dagman_text = textwrap.dedent(f'''\
        JOB gen_parameters {submit_gen_parameters_path}
        JOB submit_chunks_train {submit_chunks_train_path}
        JOB submit_chunks_test {submit_chunks_test_path}
        JOB submit_merge {submit_merge_path}
        JOB compress {submit_compress_path}

        PARENT gen_parameters CHILD submit_chunks_test
        PARENT gen_parameters CHILD submit_chunks_train
        PARENT submit_chunks_train CHILD submit_merge
        PARENT submit_chunks_test CHILD submit_merge
        PARENT submit_merge CHILD compress

        RETRY submit_chunks_train 1
        RETRY submit_chunks_test 1
        ''')

    scripts_dir = Path(submit_gen_parameters_path).resolve().parent
    dagman_path = scripts_dir/'simulation_workflow.dag'
    with open(dagman_path, 'w', encoding='utf-8') as dagman_file:
        dagman_file.write(dagman_text)
    return dagman_path
