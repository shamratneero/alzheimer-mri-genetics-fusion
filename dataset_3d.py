"""
PyTorch Dataset for the OASIS-3 Alzheimer's fusion model.

Loads cached, preprocessed 3D volumes (.npy, 160^3, already skull-stripped
and z-scored) and pairs each with:
  - label            : 0 = CN, 1 = AD
  - APOE e4 count     : 0, 1, or 2  (genetics branch input)
  - age at scan       : normalised  (clinical branch input, optional)

Volumes are downsampled at load time to TARGET_SIZE (default 128) so the
same 160^3 cache can be reused at different training resolutions without
re-preprocessing.
"""
import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class OASIS3Dataset(Dataset):
    def __init__(self, cohort_df, preprocessed_dir, target_size=128,
                 use_age=True, augment=False, extended_clinical=False):
        """
        cohort_df       : subset of oasis3_cohort.csv rows for this split
        preprocessed_dir: folder containing <subject>.npy files
        target_size     : cube side length fed to the model
        use_age         : include age_at_scan as a clinical feature
        augment         : apply light 3D augmentation (train split only)
        extended_clinical : use the 5-feature clinical vector
                            (APOE e4 count, age, sex, education, SES) instead
                            of the 2-feature one. The 2-feature version is
                            weaker than the demographics logistic-regression
                            baseline (0.781 vs 0.826), so the 5-feature version
                            is what the imaging model should be compared
                            against to avoid understating the non-imaging
                            baseline.
        """
        self.df = cohort_df.reset_index(drop=True)
        self.dir = preprocessed_dir
        self.target_size = target_size
        self.use_age = use_age
        self.augment = augment
        self.extended_clinical = extended_clinical

        # normalise age using stats from THIS dataframe's own distribution
        # (caller should pass train-set stats for val/test - see build_splits)
        self.age_mean = self.df["age_at_scan"].astype(float).mean()
        self.age_std = self.df["age_at_scan"].astype(float).std()

        # education and SES are continuous-ish integers and need scaling too;
        # sex is binary (coded 1/2 in OASIS-3) and is mapped to 0/1 instead.
        # Same leakage rule as age: callers pass train-split stats for val/test.
        self.educ_mean = self.df["education"].astype(float).mean()
        self.educ_std = self.df["education"].astype(float).std()
        self.ses_mean = self.df["ses"].astype(float).mean()
        self.ses_std = self.df["ses"].astype(float).std()

    def set_age_norm(self, mean, std):
        """Override age normalisation stats (use train-set stats for val/test)."""
        self.age_mean, self.age_std = mean, std

    def set_clinical_norm(self, stats):
        """Override all clinical normalisation stats at once (train-split stats).

        stats: dict with keys age_mean, age_std, educ_mean, educ_std,
               ses_mean, ses_std.
        """
        self.age_mean = stats["age_mean"]
        self.age_std = stats["age_std"]
        self.educ_mean = stats["educ_mean"]
        self.educ_std = stats["educ_std"]
        self.ses_mean = stats["ses_mean"]
        self.ses_std = stats["ses_std"]

    def get_clinical_norm(self):
        """Return this split's normalisation stats, to hand to val/test sets."""
        return {
            "age_mean": self.age_mean, "age_std": self.age_std,
            "educ_mean": self.educ_mean, "educ_std": self.educ_std,
            "ses_mean": self.ses_mean, "ses_std": self.ses_std,
        }

    @property
    def n_clinical_features(self):
        if self.extended_clinical:
            return 5
        return 2 if self.use_age else 1

    def __len__(self):
        return len(self.df)

    def _load_volume(self, subject):
        path = os.path.join(self.dir, f"{subject}.npy")
        vol = np.load(path)                                   # (160,160,160) float32
        t = torch.from_numpy(vol).unsqueeze(0)                # (1,160,160,160)

        if t.shape[-1] != self.target_size:
            t = F.interpolate(t.unsqueeze(0), size=(self.target_size,) * 3,
                              mode="trilinear", align_corners=False).squeeze(0)

        if self.augment:
            t = self._augment(t)

        return t

    def _augment(self, t):
        """Label-preserving 3D augmentation: flip + small translation + small
        rotation + intensity scale/shift + light gaussian noise.

        All ops preserve anatomy/label; angles and shifts are kept small so
        volumes stay physiologically plausible (this is brain MRI, not a
        natural-image classification task)."""
        if torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[1])           # left-right flip (sagittal axis)

        if torch.rand(1).item() < 0.5:
            shifts = torch.randint(-4, 5, (3,)).tolist()   # +/- 4 voxels per axis
            t = torch.roll(t, shifts=shifts, dims=[1, 2, 3])

        if torch.rand(1).item() < 0.5:
            axis = torch.randint(0, 3, (1,)).item()
            angle_deg = (torch.rand(1).item() * 2 - 1) * 10.0   # +/- 10 degrees
            t = self._rotate3d(t, angle_deg, axis)

        if torch.rand(1).item() < 0.4:
            scale = 0.9 + torch.rand(1).item() * 0.2    # 0.9 - 1.1
            shift = (torch.rand(1).item() * 2 - 1) * 0.05
            t = t * scale + shift

        if torch.rand(1).item() < 0.3:
            t = t + torch.randn_like(t) * 0.02     # small gaussian noise

        return t

    @staticmethod
    def _rotate3d(t, angle_deg, axis):
        """Rotate a (1, D, H, W) volume by angle_deg around one of the 3 axes
        using an affine grid (trilinear resample, zero-padded border)."""
        angle = math.radians(angle_deg)
        cos, sin = math.cos(angle), math.sin(angle)

        if axis == 0:      # rotate in the H-W plane
            R = torch.tensor([[1, 0, 0], [0, cos, -sin], [0, sin, cos]])
        elif axis == 1:    # rotate in the D-W plane
            R = torch.tensor([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
        else:               # rotate in the D-H plane
            R = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

        theta = torch.cat([R, torch.zeros(3, 1)], dim=1).unsqueeze(0).float()  # (1,3,4)
        grid = F.affine_grid(theta, size=(1, 1, *t.shape[1:]), align_corners=False)
        rotated = F.grid_sample(t.unsqueeze(0), grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        return rotated.squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        subject = row["subject"]

        volume = self._load_volume(subject)

        label = torch.tensor(int(row["label"]), dtype=torch.long)

        apoe_e4 = torch.tensor(float(row["apoe_e4_count"]), dtype=torch.float32)

        if self.extended_clinical:
            age_norm = (float(row["age_at_scan"]) - self.age_mean) / (self.age_std + 1e-6)
            # OASIS-3 codes sex as 1/2; map to 0/1 so it enters on a comparable
            # scale to the z-scored continuous features rather than as 1-vs-2.
            sex = float(row["sex"]) - 1.0
            educ_norm = (float(row["education"]) - self.educ_mean) / (self.educ_std + 1e-6)
            ses_norm = (float(row["ses"]) - self.ses_mean) / (self.ses_std + 1e-6)
            clinical = torch.tensor(
                [apoe_e4, age_norm, sex, educ_norm, ses_norm], dtype=torch.float32)
        elif self.use_age:
            age = float(row["age_at_scan"])
            age_norm = (age - self.age_mean) / (self.age_std + 1e-6)
            clinical = torch.tensor([apoe_e4, age_norm], dtype=torch.float32)
        else:
            clinical = torch.tensor([apoe_e4], dtype=torch.float32)

        return {
            "volume": volume,          # (1, D, H, W)
            "clinical": clinical,      # (n_clinical_features,)
            "label": label,            # scalar
            "subject": subject,        # for debugging / error analysis
        }


def build_splits(cohort_csv, test_size=0.2, val_size=0.1, seed=42):
    """
    Patient-level stratified split. Returns train/val/test DataFrames.
    Since each subject in oasis3_cohort.csv already appears exactly once
    (one scan per subject), a normal stratified split is patient-safe.
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(cohort_csv)
    df["label"] = df["label"].astype(int)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val["label"], random_state=seed)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    COHORT = "oasis3_cohort.csv"
    PRE_DIR = r"D:\alhseimer\preprocessed"

    train_df, val_df, test_df = build_splits(COHORT)
    print(f"Train: {len(train_df)}  ({train_df['label'].sum()} AD / {(train_df['label']==0).sum()} CN)")
    print(f"Val:   {len(val_df)}  ({val_df['label'].sum()} AD / {(val_df['label']==0).sum()} CN)")
    print(f"Test:  {len(test_df)}  ({test_df['label'].sum()} AD / {(test_df['label']==0).sum()} CN)")

    train_ds = OASIS3Dataset(train_df, PRE_DIR, target_size=128, augment=True)
    val_ds = OASIS3Dataset(val_df, PRE_DIR, target_size=128, augment=False)
    val_ds.set_age_norm(train_ds.age_mean, train_ds.age_std)   # use TRAIN stats

    print(f"\nTesting one sample from train set...")
    sample = train_ds[0]
    print(f"  volume shape   : {sample['volume'].shape}")
    print(f"  clinical shape : {sample['clinical'].shape}  -> {sample['clinical']}")
    print(f"  label          : {sample['label'].item()}")
    print(f"  subject        : {sample['subject']}")

    print(f"\nVolume stats: min={sample['volume'].min():.3f}  "
          f"max={sample['volume'].max():.3f}  mean={sample['volume'].mean():.3f}")

    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch shapes: volume={batch['volume'].shape}  "
          f"clinical={batch['clinical'].shape}  label={batch['label'].shape}")