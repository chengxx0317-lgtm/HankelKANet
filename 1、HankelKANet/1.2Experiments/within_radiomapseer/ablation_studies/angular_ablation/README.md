# Angular Ablation

This folder contains the angular encoding ablation experiments.

The experiments investigate the influence of Tx-centered angular Hankel encoding by introducing an additional angular coordinate map.

The input contains:

- Building layout map
- Normalized radial distance map
- Tx-centered angular map

## Ablation Settings

Different angular orders are evaluated:

```
|n| <= 1
|n| <= 2
|n| <= 3
```

Modify the variable:

```
ANGULAR_MAX_ORDER
```

in:

```
train_angular_ablation.py
```

to select different angular configurations.

## Dataset

Experiments are conducted on the RadioMapSeer dataset:

https://radiomapseer.github.io/

Please modify the dataset path in:

```
train_angular_ablation.py
```

before running.

## Training

Run:

```bash
python train_angular_ablation.py
```

The trained models, logs, and evaluation results will be saved automatically.
