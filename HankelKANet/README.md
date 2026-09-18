# HankelKANet Main Experiment

This folder contains the implementation of the proposed HankelKANet
model for path loss estimation.


------------------------------------------------------------------------

## Dataset Preparation

The main experiments are conducted on the RadioMapSeer dataset.

Dataset link:

https://radiomapseer.github.io/


Modify the dataset path in:

    loaders.py

------------------------------------------------------------------------

## Data Split

The dataset is split by environment:

-   Training: 500 environments (40,000 radio maps)
-   Validation: 100 environments (8,000 radio maps)
-   Testing: 100 environments (8,000 radio maps)

The input consists of:

-   Building layout map
-   Normalized transmitter-distance map

with resolution:

    256 × 256

------------------------------------------------------------------------

## Training

Run:

``` bash
python train_main.py
```

Default training settings:

    Optimizer: AdamW
    Learning rate: 4e-3
    Scheduler: CosineAnnealingLR

    Batch size: 32
    Epochs: 250

    Hankel orders: {0,1}

    κ range: [0.1,10]
    κ initialization: π

The trained model and training logs will be saved automatically.

------------------------------------------------------------------------

## Environment

Recommended environment:

    Python >= 3.10
    PyTorch >= 2.0
    CUDA >= 12.0

Install dependencies:

``` bash
pip install -r requirements.txt
```
