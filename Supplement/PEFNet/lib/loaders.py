import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms
import  warnings
import os
from scipy import special

warnings.filterwarnings('ignore')


def compute_Einc(tx_mask, f=5.9e9, dx=1, normalize=True):

    c = 3e8
    k = 2 * np.pi * f / c
    h, w = tx_mask.shape
    # 发射机位置
    y0, x0 = np.where(tx_mask > 0)
    if len(x0) == 0:  # fallback
        x0, y0 = [w // 2], [h // 2]
    y0, x0 = int(y0[0]), int(x0[0])
    # 坐标栅格
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) * dx + 1e-8

    H0 = special.hankel2(0, k * r)  # H0^(2) = J0 - j Y0
    Einc = -1j / 4 * H0  #
    Einc_real = np.real(Einc)
    Einc_imag = np.imag(Einc)
    Einc_real_raw = Einc_real.copy()
    Einc_imag_raw = Einc_imag.copy()

    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) * dx + 1e-6



    # 归一化
    if normalize:
        eps = 1e-9
        Einc_real = (Einc_real - Einc_real.min()) / (Einc_real.max() - Einc_real.min() + eps)
        Einc_imag = (Einc_imag - Einc_imag.min()) / (Einc_imag.max() - Einc_imag.min() + eps)

    return Einc_real, Einc_imag, Einc_real_raw, Einc_imag_raw


class Dataset_RadioMapSeer(Dataset):
    def __init__(self, phase="train",
                 dir_dataset='/',
                 numTx=80,
                 img_size=256,
                 f=5.9e9,
                 dx=0.01,#0.01
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
            self.map_inds = all_inds[:500]#[:500]
        elif phase == "val":
            self.map_inds = all_inds[500:600]
        else:
            self.map_inds = all_inds[600:]

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

        Einc_real, Einc_imag, Einc_real_raw, Einc_imag_raw = compute_Einc(tx, f=self.f, dx=self.dx)
        inputs = np.stack([bld, tx, Einc_real, Einc_imag], axis=2)
        E_inc_raw = np.stack([Einc_real_raw, Einc_imag_raw], axis=2)  # 未归一化的 E_inc


        if self.transform:
            inputs = self.transform(inputs).float()
            E_inc_raw = self.transform(E_inc_raw).float()  # 转换为张量
            gain = self.transform(gain).float()
        sample_id = f"{map_ind}_{idxc}" # 返回 inputs 和 label
        return [inputs, gain, E_inc_raw,sample_id ]  # 返回 inputs, gain 和未归一化的 E_inc
     # —— 检查通道范围 ——

        #return [inputs, gain]