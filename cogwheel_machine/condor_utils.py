"""HTCondor submission utilities."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path


def setup_condor_sub(stem, script_or_module, submit=False,
                     overwrite=False, **submit_kwargs):
    """
    Set up HTCondor submission and executable files.

    Suitable for simple jobs that run a single Python script.

    Parameters
    ----------
    stem : os.PathLike
        Generated files will be of the form `stem.{sub,sh,out,err,log}`.

    script_or_module : os.PathLike, str
        Either a path to a Python script (ending in '.py'), or a module
        name (e.g. 'my_package.my_module') to be executed.

    submit : bool
        If True, actually run `condor_submit` to submit the job. Else
        (default), simply create the submission file.

    overwrite : bool
        If False (default), raise an error if the submission or
        executable files already exist.

    **submit_kwargs
        Additional keyword arguments to be included in the submit file.
        These will be formatted as '{key} = {value}' pairs.
        For example:
        ``arguments='arg1 --arg2', request_cpus=4, request_disk='1G'``.

    Returns
    -------
    pathlib.Path
        Path to the generated submit file.
    """
    if str(script_or_module).endswith('.py'):  # script
        if not Path(script_or_module).exists():
            raise FileNotFoundError(f"{script_or_module} does not exist")
        execution_line = f'{sys.executable} {script_or_module} $@'
    else:  # module
        execution_line = f'{sys.executable} -m {script_or_module} $@'

    stem = Path(stem)
    submit_path = stem.with_suffix('.sub')
    executable_path = stem.with_suffix('.sh')

    # Path to the conda environment's lib/ directory (for shared libraries)
    env_lib = Path(sys.executable).resolve().parents[1]/'lib'

    kwarg_lines = """
        """.join(f'{key} = {value}' for key, value in submit_kwargs.items())

    submit_text = textwrap.dedent(f"""\
        executable = {executable_path}

        {kwarg_lines}

        output = {stem}.out
        error = {stem}.err
        log = {stem}.log

        queue
        """)

    executable_text = textwrap.dedent(f"""\
        #!/bin/bash
        export OMP_NUM_THREADS=1
        export LD_LIBRARY_PATH="{env_lib}:$LD_LIBRARY_PATH"

        set -e

        {execution_line}
        """)

    # Write the executable and submit files:
    write_executable(submit_path, submit_text, overwrite)
    write_executable(executable_path, executable_text, overwrite)

    if submit:
        subprocess.run(['condor_submit', str(submit_path)], check=True)
        print(f'Submitted job to HTCondor: {submit_path}')

    return submit_path


def write_executable(path, text, overwrite=False):
    """
    Write a text file and make it executable.

    Parameters
    ----------
    path : os.PathLike
        Path to the file to be created.

    text : str
        Text to be written to the file.

    overwrite : bool, optional
        If True, overwrite the file if it already exists. Default is
        False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f'{path} already exists, pass `overwrite=True` to overwrite')

    os.makedirs(path.parent, exist_ok=True)

    with open(path, 'w+', encoding='utf-8') as file:
        file.write(text)
        file.seek(0)
    path.chmod(0o755)
