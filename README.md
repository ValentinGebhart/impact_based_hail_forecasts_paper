# User-tailored impact-based hail forecasts in Switzerland

This repository contains the code used to derive the results of the paper: Gebhart V., Schmid T., Mühlhofer E., Villiger L., Bresch D. N. (2025) User-tailored impact-based hail forecasts in Switzerland, (not yet submitted).


Publication status: not yet submitted

Contact: Valentin Gebhart (valentin.gebhart@usys.ethz.ch)

## Content

### ipynb notebooks

Jupyter notebooks to reproduce figures and statistics that appear in the paper. Before running `impact_based_forecasts.ipynb` for the first time, you need to first run `impact_function_calibration.ipynb` to calibrate the impact function.

### python files

Contains several utility functions for data processing, calibration, visualization. These functions represent a subset of all utility functions used in the damage modelling part of the [scClim project](https://scclim.ethz.ch/), see also https://github.com/timschmi95/crowdsourced_paper.

## Requirements
Requires:
* Python 3.11+ environment 
* _CLIMADA_ repository version 6.*:
        https://wcr.ethz.ch/research/climada.html
        https://github.com/CLIMADA-project/climada_python
* Hail forecast data from MeteoSwiss (COSMO or ICON models).
* Shape data for Swiss cantons and warning regions.
* Exposure and damage data for the calibration. The hail damage data used in the paper are not public and only available within the [scClim](https://scclim.ethz.ch/) project.
