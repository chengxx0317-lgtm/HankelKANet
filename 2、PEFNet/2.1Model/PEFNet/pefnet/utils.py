import json
import os
import random
from pathlib import Path
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent


def load_config():
    return json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def set_seed(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def worker_seed(worker_id):
    value = torch.initial_seed() % (2 ** 32)
    np.random.seed(value)
    random.seed(value)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)
