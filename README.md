# `labrador`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Installation
### Clone repository:
```bash
git clone git@github.com:jroulet/labrador.git
```

### Create environment:
```bash
conda create -n ENVIRONMENT_NAME pip
conda activate ENVIRONMENT_NAME
```
(replace `ENVIRONMENT_NAME` by a name of your choice, e.g. `labrador`.)

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
lab-generate-data-htcondor RUNDIR --submit-arg accounting_group=ACCOUNTING_GROUP
```

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
