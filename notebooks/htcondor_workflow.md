# HTCondor submission

This page illustrates how to generate parameters and training/test data using the HTCondor scheduler.

## 1. Set up run directory and configuration file
Let us make a run directory with a copy of the example `data_config.py` in it.

```bash
lab-setup-rundir-and-priordir {parentdir}
```

where `{parentdir}` is a directory where all the data will be stored.
This command will print the `rundir` and `priordir`, which are useful for later. After this you can edit the new config files (`{rundir}/data_config.py` and `{priordir}/prior_config.py`) as needed (to specify the size of the training set, etc).

## 2. Generate the training and test sets

```bash
lab-generate-data-htcondor {priordir} --submit-arg accounting_group={accounting_group}
```

where `{accounting_group}` is a string (e.g. `'ligo.dev.o4.cbc.pe.lalinference'`).

---

The remaining steps are not yet implemented for HTCondor. Refer to the normal `workflow.ipynb`.

Anyways, by now the most CPU-intensive parts have been completed.