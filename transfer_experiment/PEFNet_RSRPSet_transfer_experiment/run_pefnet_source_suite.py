from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-root", type=Path, default=Path("./pefnet_source_runs"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).with_name("train_pefnet_source.py").resolve()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for i, seed in enumerate(seeds, 1):
        cmd = [
            args.python,
            str(script),
            "--dataset-root",
            args.dataset_root,
            "--seed",
            str(seed),
            "--output-root",
            str(args.output_root),
            *args.extra,
        ]
        print(f"[{i}/{len(seeds)}] {subprocess.list2cmdline(cmd)}")
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
