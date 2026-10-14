import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import RadioMapSeer
from .model import PEFNet
from .utils import ROOT, load_checkpoint, set_seed, write_json


def evaluate(checkpoint_path, data_root=None):
    saved = load_checkpoint(checkpoint_path, 'cpu')
    set_seed(int(saved['seed']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings = saved['settings']
    dataset = RadioMapSeer(data_root or settings['data_root'], 'test')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=settings['num_workers'], pin_memory=device.type == 'cuda')
    model = PEFNet().to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    factor = settings['db_max'] - settings['db_min']
    sums = {'MAE_dB': 0.0, 'RMSE_dB': 0.0, 'R2': 0.0}
    with torch.inference_mode():
        for batch in loader:
            prediction = model(batch['inputs'].to(device))
            label = batch['target'].to(device)
            mask = batch['outdoor'].to(device)
            prediction = prediction[mask].double()
            label = label[mask].double()
            if prediction.numel() == 0:
                raise ValueError(f"No outdoor pixels in map: {batch['id'][0]}")
            residual = prediction - label
            mse = residual.square().mean()
            variance = (label - label.mean()).square().sum()
            if variance <= 1e-12:
                raise ValueError(f"Undefined R2 for map: {batch['id'][0]}")
            sums['MAE_dB'] += residual.abs().mean().item() * factor
            sums['RMSE_dB'] += math.sqrt(mse.item()) * factor
            sums['R2'] += (1.0 - residual.square().sum() / variance).item()
    count = len(dataset)
    with Path(checkpoint_path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    result = {
        'samples': count,
        'MAE_dB': sums['MAE_dB'] / count,
        'RMSE_dB': sums['RMSE_dB'] / count,
        'R2': sums['R2'] / count,
        'seed': saved['seed'],
        'checkpoint_sha256': digest,
        'evaluation': 'Outdoor pixels, per-map metrics, arithmetic mean over maps',
    }
    output = Path(checkpoint_path).parent / 'test_metrics.json'
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--data-root', default=None)
    args = parser.parse_args()
    evaluations = [evaluate(path, data_root=args.data_root) for path in args.checkpoints]
    if len(evaluations) not in (1, 5):
        raise ValueError('Evaluate one trained model or all five independent seeds')
    if len(evaluations) == 5:
        if sorted(x['seed'] for x in evaluations) != [0, 1, 2, 3, 4]:
            raise ValueError('The five-run summary requires exactly one checkpoint for each seed 0–4')
        summary = {'runs': 5}
        for metric in ('MAE_dB', 'RMSE_dB', 'R2'):
            values = [result[metric] for result in evaluations]
            summary[metric] = {'mean': statistics.mean(values), 'std': statistics.stdev(values)}
        write_json(ROOT / 'runs' / 'evaluation_summary.json', summary)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
