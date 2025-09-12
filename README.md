# `labrador`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Installation
### Clone repository:
```bash
git clone git@github.com:jroulet/labrador.git
```

### Create environment:
```bash
conda create -n ENVIRONMENT_NAME pip cogwheel-pe sbi -c conda-forge
conda activate ENVIRONMENT_NAME
```
(replace `ENVIRONMENT_NAME` by a name of your choice, e.g. `labrador`.)

> Note: it's better to install those packages with `conda` rather than `pip`, at least in the LDG computers.

### Install:
```bash
cd labrador
pip install -e .
```

## Usage

See `notebooks/workflow.ipynb` or use the cheatsheet below.

## Cheatsheet

### 1. Create and populate `RUNDIR` (uses HTCondor)
```bash
lab-setup-rundir PARENTDIR
lab-generate-data-htcondor RUNDIR --submit-arg accounting_group=ACCOUNTING_GROUP --submit
```
> Note: this submits a `.dag` file that in turn orchestrates several `.sub` files. If you get a crash due to insufficient resources, you may adjust the requests in the corresponding `.sub`, delete from the `.dag` those jobs that have already succeeded, delete the `.rescue` file, and resubmit the `.dag` with `condor_submit_dag DAGMAN_PATH`.

### 2. Create and populate `RESCALERDIR` (uses GPU)
```bash
lab-setup-rescalerdir PRIORDIR
python -m labrador.rescaling RESCALERDIR
```

### 3. Create and populate `SBIDIR` (uses GPU)
```bash
lab-setup-sbidir RESCALERDIR
python -m labrador.training SBIDIR
```

### 4. Create and populate `UNFOLDERDIR`
```bash
lab-setup-unfolderdir RESCALERDIR
python -m labrador.unfolding UNFOLDERDIR
```
