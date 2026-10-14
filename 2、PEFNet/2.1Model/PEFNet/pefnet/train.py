import argparse
import json
import time
from pathlib import Path
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from .data import RadioMapSeer, scenario_partitions
from .model import PEFNet
from .utils import load_config, write_json, set_seed, worker_seed, load_checkpoint, ROOT


def checkpoint(model, optimizer, scheduler, epoch, best, stale, generator, settings, seed):
    return {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'best': best,
        'stale': stale,
        'generator_state': generator.get_state(),
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'numpy_rng': (np.random.get_state()[0], np.random.get_state()[1].tolist(), int(np.random.get_state()[2]), int(np.random.get_state()[3]), float(np.random.get_state()[4])),
        'python_rng': random.getstate(),
        'settings': settings,
        'seed': seed,
    }


def restore(checkpoint_data, model, optimizer, scheduler, generator):
    model.load_state_dict(checkpoint_data['model'], strict=True)
    optimizer.load_state_dict(checkpoint_data['optimizer'])
    scheduler.load_state_dict(checkpoint_data['scheduler'])
    generator.set_state(checkpoint_data['generator_state'])
    torch.set_rng_state(checkpoint_data['torch_rng'].cpu())
    if torch.cuda.is_available() and checkpoint_data['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(checkpoint_data['cuda_rng'])
    state = checkpoint_data['numpy_rng']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
    random.setstate(checkpoint_data['python_rng'])


def validation_mse(model, loader, device):
    model.eval()
    total = 0.0
    seen = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch['inputs'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            predictions = model(inputs)
            errors = (predictions - targets).square().mean(dim=(1, 2, 3))
            total += errors.sum().item()
            seen += len(inputs)
    return total / seen


def train(data_root, seed=0, resume=False):
    settings = load_config()
    settings['data_root'] = str(Path(data_root).resolve())
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = ROOT / 'runs' / f'seed_{seed}'
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config_used.json', settings)
    write_json(output / 'split.json', scenario_partitions())
    training = RadioMapSeer(settings['data_root'], 'train')
    validation = RadioMapSeer(settings['data_root'], 'val')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(training, batch_size=settings['batch_size'], shuffle=True, num_workers=settings['num_workers'], worker_init_fn=worker_seed, generator=generator, pin_memory=device.type == 'cuda', persistent_workers=settings['num_workers'] > 0)
    valid_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=settings['num_workers'], pin_memory=device.type == 'cuda', persistent_workers=settings['num_workers'] > 0)
    model = PEFNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['epochs'], eta_min=settings['minimum_learning_rate'])
    initial_epoch = 1
    best = float('inf')
    stale = 0
    last = output / 'last_checkpoint.pt'
    if resume:
        previous = load_checkpoint(last, device)
        if previous['settings'] != settings or previous['seed'] != seed:
            raise ValueError('Checkpoint experiment settings do not match')
        restore(previous, model, optimizer, scheduler, generator)
        initial_epoch = int(previous['epoch']) + 1
        best = float(previous['best'])
        stale = int(previous['stale'])
    metadata = {
        'seed': seed,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'torch_version': torch.__version__,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'train_samples': len(training),
        'validation_samples': len(validation),
        'precision': 'FP32',
    }
    write_json(output / 'metadata.json', metadata)
    for epoch in range(initial_epoch, settings['epochs'] + 1):
        start = time.perf_counter()
        model.train()
        sum_phy = 0.0
        sum_data = 0.0
        count = 0
        for batch in train_loader:
            inputs = batch['inputs'].to(device, non_blocking=True)
            incident = batch['incident'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, physical, data, loss = model(inputs, incident, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count += len(inputs)
            sum_phy += physical.detach().item() * len(inputs)
            sum_data += data.detach().item() * len(inputs)
        validation_loss = validation_mse(model, valid_loader, device)
        improved = validation_loss < best - 1e-6
        if improved:
            best = validation_loss
            stale = 0
            torch.save({'model': model.state_dict(), 'settings': settings, 'seed': seed, 'epoch': epoch, 'val_mse': validation_loss}, output / 'best_model.pt')
        else:
            stale += 1
        scheduler.step()
        torch.save(checkpoint(model, optimizer, scheduler, epoch, best, stale, generator, settings, seed), last)
        record = {'epoch': epoch, 'physics_loss': sum_phy / count, 'data_loss': sum_data / count, 'validation_mse': validation_loss, 'best_validation_mse': best, 'learning_rate': optimizer.param_groups[0]['lr'], 'epoch_seconds': time.perf_counter() - start}
        with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(f"seed={seed} epoch={epoch} physical={record['physics_loss']:.7f} data={record['data_loss']:.7f} val_mse={validation_loss:.7f} best={best:.7f}", flush=True)
        if stale >= settings['early_stopping_patience']:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.seed not in range(5):
        raise ValueError('Seed must be 0, 1, 2, 3 or 4')
    settings = load_config()
    train(args.data_root or settings['data_root'], seed=args.seed, resume=args.resume)


if __name__ == '__main__':
    main()
