# Checkpoint details

## Official result checkpoints

| Dataset | Filename | Size (bytes) | SHA-256 |
| --- | --- | ---: | --- |
| BraTS2020 | `DiTrust-Net-2020.pth` | 429606066 | `3642fa8892bee44782be6aae4891e4c75766450c4ec126bf0b402796a6ee8b31` |
| BraTS2021 | `DiTrust-Net-2021.pth` | 429606066 | `196320faaeea2e8ddef237c22e7b3d4f413c3c6af868fc69dd7a3f341d5f65e3` |

The local files were checked against these SHA-256 values during the repository documentation update. Baidu Netdisk share links and extraction codes will be added to the README; the current entries are placeholders.

## Placement and evaluation

Place the files in `results/` to use the commands in the [README](../README.md). If the files are in the repository root, pass their actual paths instead:

```bash
python validate.py --dataset-name BraTS2020 --data-root /path/to/BraTS2020 --ckpt DiTrust-Net-2020.pth
python validate.py --dataset-name BraTS2021 --data-root /path/to/BraTS2021 --ckpt DiTrust-Net-2021.pth
```

Verify file integrity on Linux with:

```bash
sha256sum DiTrust-Net-2020.pth DiTrust-Net-2021.pth
```

Or in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 DiTrust-Net-2020.pth, DiTrust-Net-2021.pth
```

## Recorded compatibility checks

The previous project README recorded the following checks against `DiTrustNet` for both checkpoints:

- Model keys: 1,551
- Checkpoint keys: 1,551
- Missing keys: 0
- Unexpected keys: 0
- Shape mismatches: 0
- `load_state_dict(..., strict=True)`: passed

These are retained project records. Model loading and full evaluation were not rerun during this documentation update.

## Distribution

The two official result checkpoints will be shared through Baidu Netdisk separately from the source code. The `.gitignore` excludes model weights, local datasets, caches, and training outputs from ordinary Git history.

The VMamba initialization weights are separate third-party assets. Refer to the upstream VMamba project for those files. Intermediate epoch checkpoints are not part of the official result checkpoint set.

