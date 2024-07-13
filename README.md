# `cogwheel-machine`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Installation

    conda create -n ENVIRONMENT_NAME python-lalsimulation -c conda-forge
    conda activate ENVIRONMENT_NAME
    git clone git@github.com:jroulet/cogwheel-machine.git
    cd cogwheel-machine
    pip install -e .

(replace `ENVIRONMENT_NAME` by a name of your choice, e.g. `cogwheel-machine`)

For now this code has no other dependencies, although soon we will require ML libraries.

## Roadmap

There are several lines of development that can happen more or less in parallel:

* Compressing data further e.g. with autoencoders (Joshua, Jay, Matias)
* Interfacing with SBI to predict the folded posterior (Marco)
* Training a classifier to unfold the posterior (Lucy)
* Encoding PSD information

## Usage

See `notebooks/workflow.ipynb`
