import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
import numpy as np

all_inds = np.arange(0, 700)
np.random.seed(42)
np.random.shuffle(all_inds)

train_maps = all_inds[0:500]
val_maps = all_inds[500:600]
test_maps = all_inds[600:700]

map_ind_target = 189
tx_idx = 26
numTx = 80

for phase_name, map_inds in [
    ("train", train_maps),
    ("val", val_maps),
    ("test", test_maps),
]:
    if map_ind_target in map_inds:
        idxr = list(map_inds).index(map_ind_target)
        idx = idxr * numTx + tx_idx
        print(f"{phase_name}: idxr={idxr}, idx={idx}")
    else:
        print(f"{phase_name}: map {map_ind_target} not in this split")