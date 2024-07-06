# `cogwheel-machine`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Installation

    conda create -n ENVIRONMENT_NAME cogwheel-pe
    conda activate ENVIRONMENT_NAME
    git clone git@github.com:jroulet/cogwheel-machine.git
    cd cogwheel-machine
    pip install -e .

(replace `ENVIRONMENT_NAME` by a name of your choice)

For now this code has no other dependencies, although soon we will require ML libraries.

## Roadmap

There are several lines of development that can happen more or less in parallel:

* Compressing data further e.g. with autoencoders (Joshua, Jay, Matias)
* Optimizing the coordinate system (Katerina, Javier)
* Interfacing with SBI to predict the folded posterior (Marco)
* Training a classifier to unfold the posterior (Lucy)
* Encoding PSD information

## Usage

To generate training data:

1. Copy-paste `cogwheel_machine/config.py` to an empty directory that will contain the simulations, we will refer to this directory as `{sim_dir}`. Edit the `config.py` copy if needed.

2. Run

        python -m cogwheel_machine.generate_parameters {sim_dir}

    This should make a file `{sim_dir}/simulation_parameters.feather`

3. Run

        python -m cogwheel_machine.simulation {sim_dir}

    This should make various files `{sim_dir}/*.npy`
