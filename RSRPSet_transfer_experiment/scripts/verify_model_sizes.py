from __future__ import annotations

import argparse
from pathlib import Path

import torch

from modules import HKANNet
from transfer_rsrpset import load_model_weights_strict


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-ch", type=int, default=2, help="Use 32 when checking a real checkpoint.")
    p.add_argument("--phys-ch", type=int, default=2, help="Use 32 when checking a real checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is not None and (args.base_ch != 32 or args.phys_ch != 32):
        print("[Info] Real main-model checkpoints normally require --base-ch 32 --phys-ch 32.")

    model = HKANNet(base_ch=args.base_ch, phys_ch=args.phys_ch)
    if args.checkpoint is not None:
        load_model_weights_strict(model, args.checkpoint.resolve())
        print("Strict checkpoint compatibility: PASS")
    model = model.to(args.device).eval()


    sizes = [(32, 32), (40, 80), (41, 79)]
    with torch.no_grad():
        for h, w in sizes:
            x = torch.rand(1, 2, h, w, device=args.device)
            y = model(x)
            ok = tuple(y.shape[-2:]) == (h, w)
            print(f"{h:3d}x{w:3d} -> {tuple(y.shape)} : {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError("Output size does not match input size.")


if __name__ == "__main__":
    main()
