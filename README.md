# Energy Consumption Forecast

The goal is to forecast the energy consumption of a 12-story large office building, a reference EnergyPlus 
commercial building model of the US Department of Energy (DoE) using **Regression Decision Tree**.

The total electricity consumption (in Joules) is predicted by assuming that the zone
temperatures and solar radiation, and wind speed can be well estimated in the test phase.

Training_test split (80-20%)

## Requirements

1. Python3
2. pandas
3. sklearn
4. matplotlib

(Anaconda and Jupyter notebook are suggested)

## Dataset

The dependent (forecasted) variable, i.e., the energy consumption, is stored in the consumption.csv file. To perform the forecast,
measurements of 16 zonal temperatures and outdoor wind, PV are used. 

Simulation data were collected for ten months with a 10-minute resolution and it contains 43263 observations.

- `features.csv` contains 18 predictors (contains header)
- `consumption.csv` labels - total electricity consumption [Joules] (does not contain header)


## Configuration 

#### Hyperparameters Tunning options
- `MAX_DEPTH_TUNNING_PLOT` plots the tree and MSE curves for training and testing and tree max depth (visualize optimal max_depth)
- `K_FOLD_CROSS_VALIDATION` calculates and prints the best max_depth and min_num_samples using k-fold cross validation
