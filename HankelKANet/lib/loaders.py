# loaders.py
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
import  os, warnings
import scipy.special as sp

warnings.filterwarnings('ignore')

def compute_Einc_and_rmap(tx_mask, f=5.9e9, dx=1.0, normalize=True):
    """计算 Tx 到各像素的归一化距离图 rho(x)"""

    h, w = tx_mask.shape

    # 发射机位置
    y0, x0 = np.where(tx_mask > 0)
    if len(x0) == 0:
        x0, y0 = [w // 2], [h // 2]
    y0, x0 = int(y0[0]), int(x0[0])

    # 坐标栅格
    yy, xx = np.meshgrid(
        np.arange(h),
        np.arange(w),
        indexing='ij'
    )

    # Tx-Rx 物理距离 d(x)，单位 m
    d = np.sqrt(
        ((xx - x0) * dx) ** 2 +
        ((yy - y0) * dx) ** 2
    ).astype(np.float32)

    if normalize:
        # 固定参考长度：整个地图物理对角线
        # 与 Tx 位置无关
        L_ref = np.sqrt(
            ((w - 1) * dx) ** 2 +
            ((h - 1) * dx) ** 2
        )

        if L_ref <= 0:
            raise ValueError("L_ref must be positive.")

        # dimensionless normalized distance rho(x)
        r_map = d / L_ref
    else:
        r_map = d

    return r_map.astype(np.float32)

class Dataset_RadioMapSeer(Dataset):
    def __init__(self, phase="train",
                 dir_dataset="your_training_folder",
                 numTx=80,
                 img_size=256,
                 f=5.9e9,
                 dx=1.0,
                 transform=None):

        self.phase = phase
        self.dir_dataset = dir_dataset
        self.dir_buildings = os.path.join(dir_dataset, 'png/buildings_complete/')
        self.dir_Tx = os.path.join(dir_dataset, 'png/antennas/')
        self.dir_gain = os.path.join(dir_dataset, 'gain/DPM/')
        self.numTx = numTx
        self.f = f
        self.dx = dx
        self.transform = transform or transforms.ToTensor()

        # 固定划分
        all_inds = np.arange(0, 700)
        np.random.seed(42)
        np.random.shuffle(all_inds)
        if phase == "train":
            self.map_inds = all_inds[0:500]
        elif phase == "val":
            self.map_inds = all_inds[500:600]
        else:
            self.map_inds = all_inds[600:700]

        self.img_size = img_size

    def __len__(self):
        return len(self.map_inds) * self.numTx

    def __getitem__(self, idx):
        idxr = idx // self.numTx
        idxc = idx % self.numTx
        map_ind = self.map_inds[idxr]
        name1 = f"{map_ind}.png"
        name2 = f"{map_ind}_{idxc}.png"

        bld = io.imread(os.path.join(self.dir_buildings, name1))
        tx = io.imread(os.path.join(self.dir_Tx, name2))
        gain = np.expand_dims(io.imread(os.path.join(self.dir_gain, name2)), axis=2)

        bld = np.asarray(bld).astype(np.float32)
        tx = np.asarray(tx).astype(np.float32)
        gain = np.asarray(gain).astype(np.float32)

        if bld.max() > 1.0: bld /= 255.0
        if tx.max() > 1.0:  tx /= 255.0
        if gain.max() > 1.0: gain /= 255.0

        if self.img_size != bld.shape[0]:
            bld = transform.resize(bld, (self.img_size, self.img_size), order=0, preserve_range=True)
            tx = transform.resize(tx, (self.img_size, self.img_size), order=0, preserve_range=True)
            gain = transform.resize(gain, (self.img_size, self.img_size), order=1, preserve_range=True)

        # 仅取 r_map
        r_map = compute_Einc_and_rmap(tx, f=self.f, dx=self.dx, normalize=True)

        # 2通道输入：bld, tx, r_map
        inputs = np.stack([bld,  r_map], axis=2)

        if self.transform:
            inputs = self.transform(inputs).float()  # (3, H, W)
            gain = self.transform(gain).float()  # (1, H, W)
        sample_id = f"{map_ind}_{idxc}"
        return inputs, gain ,sample_id  # 返回 inputs 和 label