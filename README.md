# `cogwheel-machine`

Combine simulation-based inference with gravitational-wave specific tricks such as relative binning, folding, and coordinate transformations, to get the best of both worlds.

## Prerequisites

Install [`cogwheel`](https://github.com/jroulet/cogwheel). 

For now this code has no other dependencies, although soon we will require ML libraries.

## Roadmap

There are several lines of development that can happen more or less in parallel:

* Compressing data further e.g. with autoencoders (Joshua, Jay, Matias)
* Optimizing the coordinate system (Katerina, Javier)
* Interfacing with SBI to predict the folded posterior (Marco)
* Training a classifier to unfold the posterior (Lucy)
* Encoding PSD information
