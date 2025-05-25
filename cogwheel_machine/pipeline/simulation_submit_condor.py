"""
Submit jobs to condor for generating a training set.

1. Generate training and test parameters.
2. Simulate data in chunks.
3. Merge chunks into a single file.
4. Compress the merged file.

This script generates a DAGMan file to organize the simulation jobs and
submits it to condor.
"""
import argparse
import textwrap
import subprocess
from pathlib import Path

from cogwheel_machine import compression, generate_parameters, simulation


def main(rundir, chunk_size=10_000):
    """
    Submit jobs to condor for generating a training set.

    1. Generate training and test parameters.
    2. Simulate data in chunks.
    3. Merge chunks into a single file.
    """
    # Generate submit files for all tasks
    submit_gen_parameters_path = generate_parameters.setup_condor_sub(rundir)
    submit_chunks_path, submit_merge_path = simulation.setup_condor_sub(
        rundir, chunk_size)
    submit_compress_path = compression.setup_condor_sub(rundir)

    # Generate DAGMan file for submitting jobs in the correct order
    dagman_path = _generate_dagman_file(submit_gen_parameters_path,
                                        submit_chunks_path,
                                        submit_merge_path,
                                        submit_compress_path)

    # Submit DAGMan
    subprocess.run(['condor_submit_dag', dagman_path], check=True)
    print(f'Submitted DAGMan file: {dagman_path}')


def _generate_dagman_file(submit_gen_parameters_path,
                          submit_chunks_path,
                          submit_merge_path,
                          submit_compress_path
                          ) -> Path:
    """
    Create DAGMan file to organize simulation jobs and return its path.

    Parameters
    ----------
    submit_gen_parameters_path : os.PathLike
        Path to the submit file for generating parameters.

    submit_chunks_path : os.PathLike
        Path to the submit file for simulating chunks.

    submit_merge_path : os.PathLike
        Path to the submit file for merging chunks.

    submit_compress_path : os.PathLike
        Path to the submit file for compressing chunks.

    Returns
    -------
    Path
        Path to the generated DAGMan file.
    """
    dagman_text = textwrap.dedent(f"""\
        JOB gen_parameters {submit_gen_parameters_path}
        JOB submit_chunks {submit_chunks_path}
        JOB submit_merge {submit_merge_path}
        JOB compress {submit_compress_path}

        PARENT gen_parameters CHILD submit_chunks
        PARENT submit_chunks CHILD submit_merge
        PARENT submit_merge CHILD compress

        RETRY submit_chunks 2
        """)

    scripts_dir = Path(submit_gen_parameters_path).resolve().parent
    dagman_path = scripts_dir/'simulation_workflow.dag'
    with open(dagman_path, 'w', encoding='utf-8') as dagman_file:
        dagman_file.write(dagman_text)
    return dagman_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
            Submit jobs to condor for generating a training set.

            1. Generate training and test parameters.
            2. Simulate data in chunks.
            3. Merge chunks into a single file.
            4. Compress the generated data.
            """)
    )
    parser.add_argument("rundir", type=str, help="Run directory")

    main(**vars(parser.parse_args()))
