# Multivariate Time Series Forecasting and Anomaly Detection Platform

A deep learning framework for multivariate time series forecasting and
industrial anomaly analysis.

## Project Overview

This project develops a deep learning based framework for multivariate
time series forecasting and anomaly detection.

The framework integrates data preprocessing, deep learning model
training, forecasting evaluation, residual analysis and anomaly
detection.

Besides benchmark time series datasets, industrial monitoring scenarios
are further investigated using equipment predictive maintenance and
industrial process monitoring datasets.

------------------------------------------------------------------------

## Features

-   Multivariate time series forecasting framework
-   LSTM, Transformer and iTransformer implementations
-   Forecasting evaluation with R², RMSE and MAE
-   Residual-based anomaly detection:
    -   3-Sigma
    -   IQR
    -   Isolation Forest
-   Industrial monitoring case studies

------------------------------------------------------------------------

## Datasets

### Benchmark Datasets

-   ETTh1
-   Electricity

### Industrial Monitoring Datasets

#### AI4I Predictive Maintenance Dataset

Industrial equipment condition monitoring scenario.

Variables include: - Air temperature - Process temperature - Rotational
speed - Torque - Tool wear

Tasks: - Equipment state forecasting - Failure anomaly detection

#### Tennessee Eastman Process (TEP)

Industrial process monitoring scenario with multivariate process
variables under normal and faulty conditions.

Tasks: - Normal process forecasting - Fault anomaly detection

------------------------------------------------------------------------

## Models

### LSTM

Recurrent neural network baseline for sequential modeling.

### Transformer

Attention-based sequence modeling architecture.

### iTransformer

Transformer variant designed for multivariate time series forecasting by
modeling variable relationships.

------------------------------------------------------------------------

## Industrial Case Studies

### AI4I Predictive Maintenance

Forecasting Performance:

  Model          R²      RMSE    MAE
  -------------- ------- ------- -------
  LSTM           0.639   37.79   15.50
  Transformer    0.621   38.73   16.39
  iTransformer   0.599   39.86   18.76

Residual analysis is further used for failure anomaly detection.

------------------------------------------------------------------------

### Tennessee Eastman Process Monitoring

Normal forecasting performance:

  Model          R²      RMSE    MAE
  -------------- ------- ------- -------
  LSTM           0.617   0.298   0.180
  Transformer    0.623   0.296   0.183
  iTransformer   0.707   0.261   0.117

iTransformer achieves the best performance in modeling complex
industrial multivariate dynamics.

Fault detection is performed using forecasting residuals and anomaly
detection methods.

------------------------------------------------------------------------

## Visualization

Industrial visualization includes:

-   Sensor trend analysis
-   Forecasting comparison
-   Residual analysis
-   Fault anomaly detection

Figures are stored in:

    results/AI4I/figures/
    results/TEP/figures/

------------------------------------------------------------------------

## Usage

Install dependencies:

``` bash
pip install -r requirements.txt
```

Train:

``` bash
python train.py --config configs/tep_itransformer.yaml
```

Predict:

``` bash
python predict.py --checkpoint path/to/model.pt --input path/to/test.csv --scaler path/to/scaler.joblib
```

Evaluate:

``` bash
python -m evaluation.evaluate_tep
```

------------------------------------------------------------------------

## Future Work

-   More industrial datasets
-   Advanced anomaly detection methods
-   Real-time monitoring deployment
