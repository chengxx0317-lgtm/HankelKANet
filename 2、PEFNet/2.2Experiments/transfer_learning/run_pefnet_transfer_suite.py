from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--source-checkpoint-pattern", required=True)
    p.add_argument("--output-root", type=Path, default=Path("./pefnet_rsrpset_results"))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--target-frequency-ghz", type=float, default=2.6)
    p.add_argument("--target-dx-m", type=float, default=5.0)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).with_name("pefnet_transfer_rsrpset.py").resolve()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    budgets = ["1pct", "5pct", "10pct", "20pct"]

    commands = []
    for seed in seeds:
        source_ckpt = args.source_checkpoint_pattern.format(seed=seed)

        common = [
            "--prepared-root",
            str(args.prepared_root),
            "--split-json",
            str(args.split_json),
            "--seed",
            str(seed),
            "--output-root",
            str(args.output_root),
            "--target-frequency-ghz",
            str(args.target_frequency_ghz),
            "--target-dx-m",
            str(args.target_dx_m),
        ]

        commands.append([
            args.python,
            str(script),
            "--mode",
            "zero-shot",
            "--budget",
            "0pct",
            "--source-checkpoint",
            source_ckpt,
            *common,
            *args.extra,
        ])

        for budget in budgets:
            commands.append([
                args.python,
                str(script),
                "--mode",
                "finetune",
                "--budget",
                budget,
                "--source-checkpoint",
                source_ckpt,
                *common,
                *args.extra,
            ])
            commands.append([
                args.python,
                str(script),
                "--mode",
                "scratch",
                "--budget",
                budget,
                *common,
                *args.extra,
            ])

        commands.append([
            args.python,
            str(script),
            "--mode",
            "scratch",
            "--budget",
            "100pct",
            *common,
            *args.extra,
        ])

    for i, cmd in enumerate(commands, 1):
        print(f"[{i:02d}/{len(commands):02d}] {subprocess.list2cmdline(cmd)}")
        if args.execute:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
