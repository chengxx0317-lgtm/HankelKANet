# HankelKAN: Hankel-based Kolmogorov–Arnold Network for Radio Map Prediction

This repository provides the implementation of HankelKAN for radio map prediction, including the proposed model, baseline methods, ablation studies, stratified evaluation, and cross-dataset transfer experiments.

## 1. Overview

HankelKAN integrates Hankel-based feature encoding with Kolmogorov–Arnold Network (KAN) structures to capture complex spatial propagation patterns and improve radio map prediction.

## 2. Environment

The experiments are implemented using Python and PyTorch.

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA-enabled GPU recommended

Install the required dependencies using:

`pip install -r requirements.txt`

## 3. Repository Structure

- **`1、 HankelKANet/`**
  - `1.1Model/`: Implementation of the proposed model.
  - `1.2Experiments/within_radiomapseer/`: Ablation studies and stratified evaluation.
  - `1.2Experiments/within_huawei_rsrpset/`: Within-dataset experiments on Huawei RSRPSet.
  - `1.2Experiments/transfer_learning/`: Zero-shot evaluation and fine-tuning experiments.

- **`2、 PEFNet/`**
  - `2.1Model/`: Implementation of the PEFNet baseline.
  - `2.2Experiments/`: Experiments on RadioMapSeer, Huawei RSRPSet, and cross-dataset transfer learning.

- **`3、 Other_baselines/`**
  - Implementations of ANN, DTR, KNN, RadioNet, TR36873, and TR38901.

## 4. Datasets

Experiments are conducted on two datasets:

- **RadioMapSeer:** Used for model training, baseline comparisons, ablation studies, and stratified evaluation.
- **Huawei RSRPSet:** A real-world radio propagation measurement dataset used for within-dataset evaluation and cross-dataset transfer learning. The measurements are processed into 40 × 80 radio maps with a fixed cell-level training, validation, and test split.

The datasets are not included in this repository.

## 5. Running Experiments

The training and evaluation scripts are organized into three main categories:

- **HankelKANet:** Includes experiments on RadioMapSeer and Huawei RSRPSet, ablation studies, stratified performance evaluation, and cross-dataset transfer learning.
- **PEFNet:** Includes source-domain training, within-dataset evaluation, and transfer experiments on RadioMapSeer and Huawei RSRPSet.
- **Other Baselines:** Includes implementations of additional comparison methods for radio map prediction.

For Huawei RSRPSet, the experimental configurations include training from scratch, zero-shot evaluation, and fine-tuning with limited target-domain data.

Before running the experiments, configure the dataset paths, data splits, pretrained checkpoints (if required), and Python module search paths according to the local environment. Please refer to the corresponding scripts for available command-line arguments.

## 6. Pretrained Models

Datasets and pretrained model weights are not included in this repository. The provided implementations can be used to train and evaluate the models using the corresponding experimental configurations.