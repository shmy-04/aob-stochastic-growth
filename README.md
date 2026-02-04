# aob-stochastic-growth
Single-cell–based analysis and stochastic growth modeling of ammonia-oxidizing bacteria

## Purpose
This repository contains analysis and simulation code used in the manuscript  
"Environment-responsive single-cell growth dynamics drive stochastic and deterministic population establishment in ammonia-oxidizing bacteria".

## Contents
- `0_rawdata/`: single-cell trajectory data and time-series data of cell biomass for each FOV and strain.
- `1_fit_summarize_rawdata/`: fit single-cell trajectory and biomass data to exponential curves and export processed data.
- `2_graph_summarize_data/`: generate plots summarizing single-cell behavior.
- `3_regression/`: perform regression analysis, KDE estimation, and random forest modeling to reproduce single-cell behavior.
- `4_simulation_continuous/`: simulation and validation code for continuous conditions (microfluidics).
- `5_simulation_batch/`: simulation of batch culture conditions and visualization of results.

## Notes
Scripts were executed using Python (>=3.11.14) and R (>=4.4.1).