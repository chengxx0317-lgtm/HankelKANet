import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from torch import nn
from .model import PEFNet
from .utils import load_checkpoint, set_seed, write_json


def count_conv_macs(model, inputs):
    accumulated = []
    handles = []

    def measure(layer, arguments, output):
        batch, channels, height, width = output.shape
        area = layer.kernel_size[0] * layer.kernel_size[1]
        accumulated.append(batch * channels * height * width * (layer.in_channels // layer.groups) * area)

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            handles.append(layer.register_forward_hook(measure))
    with torch.inference_mode():
        model(inputs)
    for handle in handles:
        handle.remove()
    return sum(accumulated)


def profile(checkpoint_path):
    saved = load_checkpoint(checkpoint_path, 'cpu')
    set_seed(int(saved['seed']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PEFNet().to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    x = torch.zeros(1, 4, 256, 256, dtype=torch.float32, device=device)
    x[:, 0, 128, 128] = 1.0
    macs = count_conv_macs(model, x)
    with torch.inference_mode():
        for _ in range(50):
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        measurements = []
        for _ in range(100):
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            measurements.append((time.perf_counter() - start) * 1000.0)
    milliseconds = statistics.mean(measurements)
    result = {
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
        'conv2d_macs': macs,
        'conv2d_flops': macs * 2,
        'flops_scope': 'Conv2d only; not complete model FLOPs',
        'network_latency_ms': milliseconds,
        'throughput_maps_s': 1000.0 / milliseconds,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'precision': 'FP32',
        'batch_size': 1,
        'shape': [256, 256],
        'warmup': 50,
        'repeats': 100,
        'preprocessing_included': False,
    }
    write_json(Path(checkpoint_path).parent / 'profile.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    profile(parser.parse_args().checkpoint)


if __name__ == '__main__':
    main()
