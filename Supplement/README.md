# Baseline Methods

This folder contains the implementations of baseline methods used for
comparison experiments.

The included methods are:

-   ANN: Artificial Neural Network baseline.
-   DTR: Decision Tree Regression baseline.
-   KNN: K-Nearest Neighbors baseline.
-   PEFNet: A neural network-based comparison method.
-   RadioNet: A radio map prediction baseline model.
-   TR36873: A propagation model based on the 3GPP TR 36.873
    specification.
-   TR38901: A propagation model based on the 3GPP TR 38.901
    specification.

Each subfolder contains the corresponding implementation and
experimental scripts for the specific comparison method.

## Directory Structure

``` text
baselines
│
├── ann
├── dtr
├── knn
├── PEFNet
├── radionet
├── tr36873
└── tr38901
```

## Running Experiments

Before running the baseline experiments, please check the following
items:

1.  Dataset configuration

Ensure that the required datasets are correctly prepared and that the
data paths in the configuration files or scripts are consistent with the
local environment.

2.  Model weights

Some baseline experiments may require pretrained weights or saved
checkpoints. Please make sure the corresponding weight files are
available before evaluation.

3.  Experimental results

The result files generated during experiments should be checked to
ensure that they correspond to the correct dataset, model configuration,
and evaluation settings.

Please refer to the scripts inside each subfolder for detailed training,
testing, and evaluation procedures.
