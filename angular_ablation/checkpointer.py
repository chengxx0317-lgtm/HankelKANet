# checkpointer.py
import torch
import os

class Checkpointer:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path

        self.start_epoch = 1
        self.step = 0
        self.best_val = float('inf')
        self.ckpt = None

    def load_if_exists(self, model, optimizer, scheduler, scaler, device):
        if not os.path.exists(self.ckpt_path):
            print("No checkpoint found, starting fresh.")
            return

        print("Resuming from checkpoint...")
        ckpt = torch.load(self.ckpt_path, map_location=device)
        self.ckpt = ckpt

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        # scheduler 可能不存在旧 ckpt 中
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print("Scheduler restored.")
        else:
            print("Warning: No scheduler state found!")

        self.start_epoch = ckpt.get("epoch", 1)
        self.step = ckpt.get("step", 0)
        self.best_val = ckpt.get("best_val", float('inf'))

        print(f"Loaded: epoch={self.start_epoch}, step={self.step}, best_val={self.best_val:.6f}")

    def save(self, epoch, step, best_val, model, optimizer, scheduler, scaler):
        state = {
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(state, self.ckpt_path)
       # print(f"Checkpoint saved ✔ (epoch={epoch}, step={step})")
