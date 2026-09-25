# HankelKAN: Hankel-based Kolmogorov-Arnold Network for Radio Map Prediction

This repository provides the implementation of HankelKAN for wireless
radio map prediction.

The repository includes the proposed model, baseline methods, ablation
studies, stratified evaluation, and transfer experiments.

## Overview

HankelKAN integrates Hankel-based feature encoding with
Kolmogorov-Arnold Network structures to improve radio map prediction by
learning complex spatial propagation patterns.

## Main Features

-   Hankel-based feature representation.
-   KAN-based neural network architecture.
-   Complete training and evaluation pipeline.
-   Comprehensive experiments:
    -   Baseline comparisons.
    -   Angular-order ablation.
    -   Hankel order ablation.
    -   Stratified evaluation.
    -   RSRPSet transfer experiments.

## Environment Installation

The experiments are implemented with Python and PyTorch.

Recommended environment:

``` text
Python >= 3.10
PyTorch >= 2.0
CUDA-enabled GPU environment
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Repository Structure

``` text
HankelKAN-release
│
├── HankelKANet
│   Main implementation of HankelKAN.
│
├── Supplement
│   Implementations of comparison methods.
│
├── angular_ablation
│   Angular-order ablation experiments.
│
├── order_ablation
│   Hankel order ablation experiments.
│
├── Stratified_comparison
│   Stratified evaluation experiments.
│
├── RSRPSet_transfer_experiment
│   Transfer experiments on the RSRPSet dataset.
│
└── requirements.txt
```

## Experiments

The corresponding implementations are provided in the following folders:

-   `HankelKANet`: Main model experiments.
-   `baselines`: Comparison methods including ANN, DTR, KNN, PEFNet,
    RadioNet, TR36873, and TR38901.
-   `angular_ablation`: Angular-order ablation studies.
-   `order_ablation`: Hankel order ablation studies.
-   `Stratified_comparison`: Evaluation under different propagation
    conditions.
-   `RSRPSet_transfer_experiment`: Transfer experiments on the RSRPSet
    dataset.

Please refer to each folder for detailed training and evaluation
procedures.

## Dataset

The datasets are not included in this repository.

Please prepare the datasets according to the instructions provided in
the corresponding folders.

## Pretrained Models

Pretrained models are not included in this repository.

The released code provides the complete implementation for training and
evaluation. Users can reproduce the experimental results by training the
model according to the provided instructions.
