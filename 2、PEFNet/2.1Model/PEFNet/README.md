# PEFNet

PyTorch implementation of PEFNet, based on *Physics-Informed Neural Networks for Path Loss Estimation by Solving Electromagnetic Integral Equations* (IEEE TWC, 2024).

## Project Structure

```text
PEFNet/
├── pefnet/
│   ├── __init__.py
│   ├── model.py
│   ├── vie.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── profile.py
│   └── utils.py
├── config.json
├── requirements.txt
├── README.md
└── .gitignore
```

## File Descriptions

| File | Description |
|---|---|
| `model.py` | Implements the dual-UNet architecture and path loss prediction. |
| `vie.py` | Implements the volume integral equation (VIE) and physics-informed loss. |
| `data.py` | Handles dataset loading, preprocessing, and data partitioning. |
| `train.py` | Performs model training, validation, and checkpoint saving. |
| `evaluate.py` | Evaluates prediction accuracy using R², MAE, and RMSE. |
| `profile.py` | Measures model complexity, inference latency, and GPU memory usage. |
| `utils.py` | Provides shared utilities for configuration, reproducibility, and checkpoint management. |
| `__init__.py` | Initializes the Python package. |
| `config.json` | Defines model parameters and experimental settings. |
| `requirements.txt` | Lists required Python dependencies. |
| `.gitignore` | Excludes temporary files and generated artifacts from version control. |
