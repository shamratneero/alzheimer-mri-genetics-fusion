"""
3D CNN + clinical fusion model for Alzheimer's classification.

Architecture:
  Imaging branch : 5-block 3D CNN (Conv3d + BatchNorm + ReLU + MaxPool),
                   channels 16 -> 32 -> 64 -> 128 -> 256, global avg pool
  Clinical branch: small MLP on APOE e4 count (+ age)
  Fusion         : concatenate both feature vectors -> classifier head

Designed to fit an 8GB GPU at 128^3 input with batch size 4-8.
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Conv3d -> BatchNorm -> ReLU -> Conv3d -> BatchNorm -> ReLU -> MaxPool.
    Two convs per block gives more representational power per downsample."""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class ImagingBranch3D(nn.Module):
    """128^3 input -> 256-d feature vector.
    128 -> 64 -> 32 -> 16 -> 8 -> 4 (spatial), channels 1->16->32->64->128->256."""
    def __init__(self, in_channels=1, base_ch=16, dropout=0.3):
        super().__init__()
        self.block1 = ConvBlock3D(in_channels, base_ch)          # 128 -> 64
        self.block2 = ConvBlock3D(base_ch, base_ch * 2)          # 64  -> 32
        self.block3 = ConvBlock3D(base_ch * 2, base_ch * 4)      # 32  -> 16
        self.block4 = ConvBlock3D(base_ch * 4, base_ch * 8)      # 16  -> 8
        self.block5 = ConvBlock3D(base_ch * 8, base_ch * 16)     # 8   -> 4
        self.gap = nn.AdaptiveAvgPool3d(1)                       # -> 1x1x1
        self.out_dim = base_ch * 16
        # Global average pooling over a 4x4x4 grid averages away most of the
        # variance, leaving features with std ~0.005 - roughly 45x weaker than
        # the clinical branch's output. LayerNorm rescales them to unit scale so
        # neither branch is privileged purely by feature magnitude, both in
        # fusion (where the larger branch would otherwise dominate the
        # concatenated vector) and across the single-branch ablations.
        self.norm = nn.LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x).flatten(1)          # (B, out_dim)
        x = self.norm(x)
        return self.dropout(x)


class ClinicalBranch(nn.Module):
    """Small MLP for tabular clinical/genetic features."""
    def __init__(self, in_features=2, hidden=16, out_dim=16, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
            # Matches the LayerNorm on the imaging branch output so both
            # branches hand the classifier features on the same scale.
            nn.LayerNorm(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    """Imaging branch + clinical branch -> concatenated -> classifier head."""
    def __init__(self, n_clinical_features=2, base_ch=16,
                 clinical_hidden=16, clinical_out=16,
                 fusion_hidden=64, n_classes=2, dropout=0.3):
        super().__init__()
        self.imaging = ImagingBranch3D(base_ch=base_ch, dropout=dropout)
        self.clinical = ClinicalBranch(in_features=n_clinical_features,
                                       hidden=clinical_hidden, out_dim=clinical_out)

        fused_dim = self.imaging.out_dim + self.clinical.out_dim

        def _make_head(in_dim):
            """Classifier head sized to its actual input width.

            Each mode gets its own head rather than zero-padding into one
            shared `fused_dim`-wide head. Reason: nn.Linear initialises weights
            with std ~ 1/sqrt(fan_in), so routing a 16-d clinical vector through
            a 272-wide head (256 slots permanently zero) starts its weights
            ~4.4x too small and systematically handicaps the clinical-only
            ablation relative to imaging-only. Sizing each head to its real
            input makes the three modes genuinely comparable."""
            return nn.Sequential(
                nn.Linear(in_dim, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, n_classes),
            )

        self.classifier = _make_head(fused_dim)                    # fusion
        self.classifier_imaging = _make_head(self.imaging.out_dim)   # imaging-only
        self.classifier_clinical = _make_head(self.clinical.out_dim) # clinical-only

    def forward(self, volume, clinical):
        img_feat = self.imaging(volume)
        clin_feat = self.clinical(clinical)
        fused = torch.cat([img_feat, clin_feat], dim=1)
        return self.classifier(fused)

    def forward_imaging_only(self, volume):
        """Imaging-only ablation: MRI features through their own head."""
        img_feat = self.imaging(volume)
        return self.classifier_imaging(img_feat)

    def forward_clinical_only(self, clinical):
        """Clinical-only ablation (APOE + age, no imaging): clinical features
        through their own correctly-sized head, so this branch is not
        disadvantaged by the fusion head's much larger fan_in."""
        clin_feat = self.clinical(clinical)
        return self.classifier_clinical(clin_feat)


if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = FusionModel(n_clinical_features=2, base_ch=16).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {n_params:,}")
    print(f"Trainable parameters : {n_trainable:,}")
    print(f"Model size (approx)  : {n_params * 4 / 1e6:.1f} MB (fp32)")

    # --- VRAM probe: one forward + backward pass at increasing batch sizes ---
    print(f"\n{'='*55}")
    print("VRAM PROBE (forward + backward pass)")
    print(f"{'='*55}")

    for batch_size in [2, 4, 6, 8, 12]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            vol = torch.randn(batch_size, 1, 128, 128, 128, device=device)
            clin = torch.randn(batch_size, 2, device=device)
            labels = torch.randint(0, 2, (batch_size,), device=device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            optimizer.zero_grad()
            out = model(vol, clin)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                print(f"  batch_size={batch_size:3d}  peak VRAM: {peak_mb:8.1f} MB")
            else:
                print(f"  batch_size={batch_size:3d}  (CPU - no VRAM tracking)")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  batch_size={batch_size:3d}  OUT OF MEMORY")
                torch.cuda.empty_cache()
                break
            else:
                raise

    print(f"\nGPU total VRAM: "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
          if device.type == "cuda" else "")