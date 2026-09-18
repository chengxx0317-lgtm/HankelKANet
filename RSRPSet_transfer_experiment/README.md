# RSRPSet Transfer Experiment

This folder contains the implementation of transfer experiments on the RSRPSet dataset.

The experiments evaluate the generalization capability of the proposed model on unseen propagation scenarios through zero-shot evaluation, fine-tuning with limited target-domain data, and training-from-scratch baselines.

## Experimental Settings

The transfer experiments include:

- Zero-shot evaluation without additional training.
- Fine-tuning with different proportions of available target-domain data.
- Training from scratch with matched data budgets for comparison.

The corresponding experimental configurations and evaluation results are provided in the results folder.

## Dataset Preparation

The processed RSRPSet data are not included in this repository.

Users should prepare the dataset according to the provided preprocessing scripts. The preprocessing pipeline generates model-ready inputs, receiver masks, and fixed train/validation/test splits.

## Running Experiments

### Zero-shot evaluation

```bash
python zero_shot_rsrpset.py
```

### Fine-tuning

Example:

```bash
python adapt_rsrpset.py --fraction 0.01 --seed 0
```

where `--fraction` specifies the proportion of target-domain training data.

### Training from scratch

Example:

```bash
python transfer_rsrpset.py --mode scratch --fraction 0.01 --seed 0
```

### Complete pipeline

The complete experimental pipeline can be executed using:

```bash
python run_transfer_suite.py
```

The trained models, logs, and evaluation results will be saved automatically.
