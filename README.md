![labrador](https://raw.githubusercontent.com/jroulet/labrador/cleanup/docs/source/_static/labrador.jpg)

`labrador` combines simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Reference
[labrador: A domain-optimized machine-learning tool for gravitational wave inference](https://arxiv.org/abs/2604.08897)

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

> Note: it's better to install those packages with `conda` rather than `pip`, at least in the LIGO Data Grid computers.

### Install:
```bash
cd labrador
pip install -e .
```

## Training

See `notebooks/workflow.ipynb` or use the cheatsheet below.

### Cheatsheet

#### 1. Create and populate `RUNDIR` (uses HTCondor)
```bash
lab-setup-rundir-and-priordir PARENTDIR
# Edit config files...
lab-generate-data-htcondor PRIORDIR --submit-arg accounting_group=ACCOUNTING_GROUP --submit
```
> Note: this submits a `.dag` file that in turn orchestrates several `.sub` files. If you get a crash due to insufficient resources, you may adjust the requests in the corresponding `.sub`, delete from the `.dag` those jobs that have already succeeded, delete the `.rescue` file, and resubmit the `.dag` with `condor_submit_dag DAGMAN_PATH`.

#### 2. Create and populate `RESCALERDIR` (uses GPU)
```bash
lab-setup-rescalerdir PRIORDIR
python -m labrador.rescaling RESCALERDIR
```

#### 3. Create and populate `SBIDIR` (uses GPU)
```bash
lab-setup-sbidir RESCALERDIR
python -m labrador.training SBIDIR
```

#### 4. Create and populate `UNFOLDERDIR`
```bash
lab-setup-unfolderdir RESCALERDIR
python -m labrador.unfolding UNFOLDERDIR
```

## Inference

See https://zenodo.org/records/19393278 for a demonstration with already-trained models.

## Troubleshooting

### If jobs are held

Diagnose with

    condor_q

Find which jobs were held:

    cd {rundir}/submission_scripts
    grep -R held

This will point to the relevant log files. Example output:

    simulation-7190000_7200000_train.log:012 (527015058.719.000) 2026-03-18 14:57:48 Job was held.
    simulation-4890000_4900000_train.log:012 (527015058.489.000) 2026-03-18 14:51:48 Job was held.
    simulation-5080000_5090000_train.log:012 (527015058.508.000) 2026-03-18 14:52:48 Job was held.

Check the logs. Two causes for eviction are

1. Insufficient resources requested (e.g. memory), in that case edit the relevant .sub file and resubmit.

2. Bad nodes.
In that case the NumShadowStarts variable will be high:

        grep -R NumShadowStarts

    Example output

        simulation-7190000_7200000_train.log:   NumShadowStarts 148 > 100.
        simulation-4890000_4900000_train.log:   NumShadowStarts 118 > 100.
        simulation-5080000_5090000_train.log:   NumShadowStarts 127 > 100.

    Reset like so:

        condor_qedit 527015058.719 NumShadowStarts 0
        condor_qedit 527015058.489 NumShadowStarts 0
        condor_qedit 527015058.508 NumShadowStarts 0

    (get the correct job numbers for your case from the `grep -R held` output). Then resubmit:

        condor_release albert.einstein


## Acknowledgements

We are grateful to Eliot Finch for designing the `labrador` logo.
