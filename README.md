# aob-stochastic-growth
Single-cell–based analysis and stochastic growth modeling of ammonia-oxidizing bacteria

## Purpose
This repository contains analysis and simulation code used in the manuscript  
"Environment-responsive individual cell growth behavior shapes stochastic and deterministic population establishment in ammonia-oxidizing bacteria".

## Contents
- `0_rawdata/`: single-cell trajectory data and time-series data of cell biomass for each FOV and strain.
- `1_fit_summarize_rawdata/`: fit single-cell trajectory and biomass data to exponential curves and export processed data.
- `2_graph_summarize_data/`: generate plots summarizing single-cell behavior.
- `3_regression/`: perform regression analysis, KDE estimation, and random forest modeling to reproduce single-cell behavior.
- `4_simulation_continuous/`: simulation and validation code for continuous conditions (microfluidics).
- `5_simulation_batch/`: simulation of batch culture conditions and visualization of results.

## Notes
Scripts were executed using Python (>=3.11.14) and R (>=4.4.1).

## How to Cite

If you use the analysis or simulation code provided in this repository, please cite:

Ikeda S. 2026. *aob-stochastic-growth*. GitHub. (https://github.com/shmy-04/aob-stochastic-growth)

## Related Publication

The analysis and simulation code provided in this repository was developed and used in the following study:

- [Ikeda et al. 2026. *mBio*](https://journals.asm.org/doi/10.1128/mbio.02345-26)