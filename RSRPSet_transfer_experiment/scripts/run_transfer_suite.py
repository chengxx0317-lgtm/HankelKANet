from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--source-checkpoint-pattern", required=True)
    p.add_argument("--output-root", type=Path, default=Path("./rsrpset_transfer_results"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).with_name("transfer_rsrpset.py").resolve()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    fractions = [0.01, 0.05, 0.10, 0.20]

    commands = []
    for seed in seeds:
        ckpt = args.source_checkpoint_pattern.format(seed=seed)

        # 0% zero-shot
        commands.append([
            args.python, str(script),
            "--mode", "zero-shot",
            "--prepared-root", str(args.prepared_root),
            "--split-json", str(args.split_json),
            "--source-checkpoint", ckpt,
            "--seed", str(seed),
            "--output-root", str(args.output_root),
            *args.extra,
        ])

        # Pretrained adaptation and matched scratch control.
        for frac in fractions:
            commands.append([
                args.python, str(script),
                "--mode", "finetune",
                "--prepared-root", str(args.prepared_root),
                "--split-json", str(args.split_json),
                "--source-checkpoint", ckpt,
                "--fraction", str(frac),
                "--seed", str(seed),
                "--output-root", str(args.output_root),
                *args.extra,
            ])
            commands.append([
                args.python, str(script),
                "--mode", "scratch",
                "--prepared-root", str(args.prepared_root),
                "--split-json", str(args.split_json),
                "--fraction", str(frac),
                "--seed", str(seed),
                "--output-root", str(args.output_root),
                *args.extra,
            ])

        # 100% real-data scratch upper reference.
        commands.append([
            args.python, str(script),
            "--mode", "scratch",
            "--prepared-root", str(args.prepared_root),
            "--split-json", str(args.split_json),
            "--fraction", "1.0",
            "--seed", str(seed),
            "--output-root", str(args.output_root),
            *args.extra,
        ])

    for i, cmd in enumerate(commands, 1):
        printable = subprocess.list2cmdline(cmd)
        print(f"[{i:02d}/{len(commands):02d}] {printable}")
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
