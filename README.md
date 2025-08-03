# `labrador`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Installation
### Clone repository:
```bash
git clone git@github.com:jroulet/labrador.git
```

### Create environment:
```bash
conda create -n ENVIRONMENT_NAME pip cogwheel-pe sbi
conda activate ENVIRONMENT_NAME
```
(replace `ENVIRONMENT_NAME` by a name of your choice, e.g. `labrador`.)

Note: it's better to install those packages with `conda` rather than `pip`, at least in the LDG computers.

### Install:
```bash
cd labrador
pip install -e .
```

## Usage

See `notebooks/workflow.ipynb`
