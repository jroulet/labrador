"""
Functions to run cogwheel, for comparison.

Create data that is compatible with both cogwheel and labrador.
"""
import argparse
import os
from pathlib import Path

import numpy as np

import cogwheel.gw_utils
import cogwheel.posterior
import cogwheel.sampling
import cogwheel.utils
import cogwheel.waveform

from . import compression, simulation, utils

EVENTS_DIRNAME = 'example_events'


def make_eventdir(priordir):
    """
    Make and return directory of the form priordir/EVENTS_DIRNAME/GW{i}.

    See Also
    --------
    main
    """
    return utils.make_unique_dir(priordir/EVENTS_DIRNAME, 'GW')


def main(eventdir, sampler_cls, run_inference=True):
    """
    Create an injection, save it, and launch a ``cogwheel`` inference.

    This will:

    1. Draw parameters from the physical prior and create
       synthetic data (constrained to satisfy the mask constraints in
       ``rundir/data_config.py``)
    2. Save the data in `eventdir` so that they are readable by
       ``cogwheel`` and ``labrador``
    3. Infer the posterior using a stochastic sampler (``cogwheel``).

    Can be used for comparing ``labrador`` against ``cogwheel``.

    Parameters
    ----------
    eventdir : os.PathLike
        (Empty) directory inside ``priordir/EVENTS_DIRNAME``, the
        injection will be created here. See :py:func:`make_eventdir`.

    sampler_cls : class, str
        Either a subclass of ``cogwheel.sampling.Sampler``, or a string
        with its name (e.g. 'Dynesty', 'Nautilus', 'PyMultiNest', ...).

    run_inference : bool
        True (default): save and run the cogwheel sampler.
        False: just save the cogwheel sampler to json.
    """
    eventdir = Path(eventdir).resolve()
    priordir, rundir = eventdir.parents[1 : 3]

    prior = utils.load_prior_config(priordir).PRIOR

    event_data, compressed_data, transform = generate_data_and_transform(
        rundir)
    event_data.eventname = eventdir.name

    transform.to_json(eventdir, basename='Transform.json')
    np.save(eventdir/utils.COMPRESSED_DATA_FILENAME, compressed_data)

    prior = _adjust_mchirp_range(prior, event_data.injection['par_dic'])
    posterior = build_cogwheel_posterior(event_data, prior)
    sampler = _get_sampler_cls(sampler_cls)(posterior)

    if run_inference:
        sampler.run(eventdir)
    else:
        sampler.to_json(eventdir)


def _adjust_mchirp_range(prior, par_dic):
    uses_mchirp = (
        'mchirp' in prior.sampled_params
        and any(par.name == 'mchirp_range' for par in prior.init_parameters())
    )
    if not uses_mchirp:
        return prior

    mchirp = cogwheel.gw_utils.m1m2_to_mchirp(par_dic['m1'], par_dic['m2'])
    mchirp_range = cogwheel.gw_utils.estimate_mchirp_range(mchirp, sigmas=10.0)
    np.clip(mchirp_range, *prior.range_dic['mchirp'], out=mchirp_range)

    return prior.reinstantiate(mchirp_range=mchirp_range)


def build_cogwheel_posterior(event_data, prior):
    """
    Build a cogwheel posterior from an event data with an injection.

    Parameters
    ----------
    event_data : cogwheel.data.EventData
        Needs to have an injection (`event_data.injection`).

    prior : cogwheel.prior.Prior
        Defines the prior and the likelihood class.

    Returns
    -------
    cogwheel.posterior.Posterior
    """
    waveform_generator = cogwheel.waveform.WaveformGenerator.from_event_data(
        event_data, event_data.injection['approximant'])

    likelihood = prior.default_likelihood_class(
        event_data=event_data,
        waveform_generator=waveform_generator,
        par_dic_0=event_data.injection['par_dic'],
        pn_phase_tol=0.05,
    )
    return cogwheel.posterior.Posterior(prior, likelihood)


def _get_sampler_cls(sampler_cls):
    if isinstance(sampler_cls, str):
        sampler_cls = next(
            cls  for cls in cogwheel.sampling.Sampler.__subclasses__()
            if cls.__name__ == sampler_cls)

    return sampler_cls


def generate_data_and_transform(rundir, prior_cls=None):
    """
    Return event data, compressed data and transform for a random
    simulated event, ensuring the mask conditions are satisfied.

    Parameters
    ----------
    rundir : os.PathLike
        Path to run directory.

    prior_cls : class
        A subclass of cogwheel.prior.Prior, to draw the parameters from.
        Defaults to the simulation prior.

    Returns
    -------
    event_data : cogwheel.data.EventData
        Contains strain data with injection and noise.

    compressed_data : numpy.ndarray
        Input to SBI posterior.

    transform : labrador.transform.TransformMixin
        Instance of the transform class that corresponds to these data.
    """
    data_config = utils.load_data_config(rundir)

    if prior_cls is None:
        prior_cls = data_config.PRIOR_CLASS

    prior = prior_cls(**data_config.PRIOR_KWARGS)
    simulator, data_preprocessor, transform_class \
        = simulation.setup_simulator(rundir)

    satisfactory = False
    while not satisfactory:
        parameters = prior.generate_random_samples(1).iloc[0]

        simulated_input = simulator.generate_data_and_reference_waveform(
            parameters)

        preprocessed_data, transform_kwargs \
            = data_preprocessor.preprocess_data(**simulated_input)

        satisfactory = _mask(
            parameters, preprocessed_data, data_config.MASK_CONDITIONS)

    simulated_input['event_data'].injection['par_dic'] = dict(
        simulated_input['event_data'].injection['par_dic'])  # Series -> dict

    compressed_data = compression.compress_data(
        rundir,
        preprocessed_data['heterodyned_data'],
        preprocessed_data['processed_coef'])

    transform = transform_class(**transform_kwargs)

    return simulated_input['event_data'], compressed_data, transform


def _mask(parameters, preprocessed_data, mask_conditions):
    """Return whether all the mask conditions are satisfied."""
    summary = parameters.copy()
    _add_snr_to_summary(summary, preprocessed_data)

    return all(logic(summary[par], value)
               for par, logic, value in mask_conditions)


def _add_snr_to_summary(summary, preprocessed_data):
    """Add keys (d_h, h_h, h0_h0, snr, snr0) inplace to `summary`."""
    # TODO unify with utils.py
    for key in 'd_h', 'h_h', 'h0_h0':
        summary[key] = np.sum(preprocessed_data[key], axis=-1)

    summary['snr'] = summary['d_h'] / np.sqrt(summary['h_h'])
    summary['snr0'] = np.sqrt(summary['h0_h0'])


def submit_condor(priordir,
                  sampler_cls,
                  request_cpus=1,
                  request_memory='1G',
                  request_disk='1G',
                  **submit_kwargs):
    """
    Submit an HTCondor job to generate simulation parameters.

    This will generate the following files::

        submission_scripts/inj_{i}/injections.{sub,sh,out,err,log}

    Parameters
    ----------
    priordir : os.PathLike
        Directory inside ``rundir``, corresponding to a physical prior.
        See :py:func:`utils.setup_priordir`.

    request_cpus, request_memory, request_disk : int or str
        Specifications in the HTCondor submit file.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do not
        pass `executable`, `output`, `error`, `log`, `args`, `queue`,
        which will be dealt with automatically.
    """
    priordir = Path(priordir).resolve()
    eventdir = make_eventdir(priordir)

    if not isinstance(sampler_cls, str):
        sampler_cls = sampler_cls.__name__

    scripts_dir = eventdir/'submission_scripts'
    os.makedirs(scripts_dir)

    submit_kwargs = {
        'submit_path': scripts_dir/'injections.sub',
        'executable': scripts_dir/'injections.sh',
        'output': scripts_dir/'injections.out',
        'error': scripts_dir/'injections.err',
        'log': scripts_dir/'injections.log',
        'args': f'{eventdir} {sampler_cls}',
        'request_cpus': request_cpus,
        'request_memory': request_memory,
        'request_disk': request_disk,
        **submit_kwargs,
    }

    cogwheel.utils.submit_condor(**submit_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Create an injection, save it, and launch a `cogwheel`
                       inference.''')
    parser.add_argument('eventdir',
                        help='Event directory, see injections.make_eventdir.')
    parser.add_argument('sampler_cls',
                        help='cogwheel.sampling.Sampler subclass.')

    main(**vars(parser.parse_args()))
