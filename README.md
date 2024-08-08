# `cogwheel-machine`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Installation
Clone repository:

    git clone git@github.com:jroulet/cogwheel-machine.git

Create environment with `cogwheel-pe`:

    conda create -n ENVIRONMENT_NAME cogwheel-pe -c conda-forge
    conda activate ENVIRONMENT_NAME

(replace `ENVIRONMENT_NAME` by a name of your choice, e.g. `cogwheel-machine`.)

Install:

    cd cogwheel-machine
    pip install -e .

## Roadmap

There are several lines of development that can happen more or less in parallel:

* Compressing data further e.g. with autoencoders (Joshua, Jay, Matias)
* Training a classifier to unfold the posterior (Lucy)
* Encoding PSD information

## Usage

See `notebooks/workflow.ipynb`
