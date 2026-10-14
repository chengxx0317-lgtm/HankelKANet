# Hankel Order Ablation

This folder contains the Hankel order ablation experiments for evaluating the influence of different Hankel basis configurations.

The network architecture and training settings are kept consistent with the main experiment. Only the Hankel order configuration is changed.

## Ablation Settings

The following Hankel order combinations are evaluated:

```
{0}
{1}
{0,1}
{0,1,2}
{0,1,2,3}
```

Modify the variable `HANKEL_ORDERS` in:

```
train_order_ablation.py
```

to select different configurations.

## Dataset

Experiments are conducted on the RadioMapSeer dataset:

https://radiomapseer.github.io/

Please modify the dataset path in:

```
train_order_ablation.py
```

before running.

## Training

Run:

```bash
python train_order_ablation.py
```

The trained models, logs, and evaluation results will be saved automatically.
