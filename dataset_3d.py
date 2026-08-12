import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

class OASIS3Dataset(Dataset):
    """
    Loads 3D T1 MRI volumes from OASIS-3.
    Expects a list of (nifti_filepath, label) pairs.
    """
    def __init__(self, samples, target_shape=(128, 128, 128), normalize=True):
        self.samples = samples          # list of (filepath, label)
        self.target_shape = target_shape
        self.normalize = normalize

    def __len__(self):
        return len(self.samples)

    def _load_and_preprocess(self, filepath):
        # Load NIfTI volume
        img = nib.load(filepath)
        volume = img.get_fdata().astype(np.float32)

        # Resize/downsample to target shape
        factors = [t / s for t, s in zip(self.target_shape, volume.shape)]
        volume = zoom(volume, factors, order=1)  # linear interpolation

        # Intensity normalization (zero mean, unit variance)
        if self.normalize:
            mean, std = volume.mean(), volume.std()
            if std > 0:
                volume = (volume - mean) / std

        # Add channel dimension: (1, D, H, W)
        volume = np.expand_dims(volume, axis=0)
        return torch.from_numpy(volume).float()

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        volume = self._load_and_preprocess(filepath)
        return volume, label


if __name__ == "__main__":
    # Placeholder test - replace with a real OASIS-3 .nii path once downloaded
    test_samples = [
        # ("path/to/some_oasis3_scan.nii.gz", 0),
    ]
    if test_samples:
        ds = OASIS3Dataset(test_samples)
        vol, label = ds[0]
        print(f"Volume shape: {vol.shape}, label: {label}")
    else:
        print("No test samples yet - add a real OASIS-3 NIfTI path to test.")