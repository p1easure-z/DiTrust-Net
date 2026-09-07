# DiTrust-Net

DiTrust-Net is a 2D multimodal brain-tumor segmentation model for BraTS 2020
and BraTS 2021. The input is a four-channel FLAIR/T1/T1ce/T2 slice, and the
output contains the WT, TC, and ET regions.

## Project structure

```text
backbone/vmamba/       VMamba backbone and CUDA/Triton operators
data/                  data loader and the original preprocessing script
kits/                  losses, metrics, logging, and scheduler
networks/Net.py        DiTrust-Net model
train.py               training entry point
trainer.py             training and validation loop
validate.py            standalone evaluation entry point
```

## Environment

The current checkpoints were verified with Python 3.10, PyTorch 2.1.1 + CUDA
12.1, and Triton 2.1.0. Install the remaining packages with:

```bash
pip install -r requirements.txt
```

The VMamba path relies on CUDA/Triton operators, so normal training and
inference require an NVIDIA GPU.

## Data layout

The code expects preprocessed NumPy files in this layout:

```text
/path/to/BraTS2020/
├── trainImage/   # [224, 224, 4]
├── trainGt/      # [224, 224], labels 0/1/2/4
├── testImage/
└── testGt/
```

Set `BRATS2020_ROOT` or `BRATS2021_ROOT` before training. Validation also
accepts `--data-root`. For preprocessing, set `BRATS_HGG_ROOT` and
`BRATS_LGG_ROOT`; the preprocessing calculations remain unchanged.

## Checkpoints

Only the following two files are official result checkpoints:

| Dataset   | File                           | SHA-256                                                      |
| --------- | ------------------------------ | ------------------------------------------------------------ |
| BraTS2020 | `results/DiTrust-Net-2020.pth` | `3642fa8892bee44782be6aae4891e4c75766450c4ec126bf0b402796a6ee8b31` |
| BraTS2021 | `results/DiTrust-Net-2021.pth` | `196320faaeea2e8ddef237c22e7b3d4f413c3c6af868fc69dd7a3f341d5f65e3` |

Both files have been checked against the current `DiTrustNet` definition:

- model keys: 1,551
- checkpoint keys: 1,551
- missing keys: 0
- unexpected keys: 0
- shape mismatches: 0
- `load_state_dict(..., strict=True)`: passed

Each checkpoint is about 410 MiB and is excluded by `.gitignore`. Publish only
these two files separately with a GitHub Release or Git LFS; do not upload the
epoch checkpoints under `results/weights/` or the obsolete files under `vis/`.

## Validation

```bash
python validate.py \
  --dataset-name BraTS2020 \
  --data-root /path/to/BraTS2020 \
  --ckpt results/DiTrust-Net-2020.pth
```

The existing complete-split reports contain these Dice scores:

| Dataset   | Slices |     WT |     TC |     ET | Average |
| --------- | -----: | -----: | -----: | -----: | ------: |
| BraTS2020 |  4,709 | 0.9145 | 0.8659 | 0.8702 |  0.8835 |
| BraTS2021 | 16,321 | 0.9326 | 0.9315 | 0.9018 |  0.9220 |

## Training

After configuring the dataset environment variable:

```bash
export BRATS2020_ROOT=/path/to/BraTS2020
python train.py --dataset_name BraTS2020 --epochs 100 --batch_size 6
```

Training outputs are written under `results/`.

## Acknowledgment

The backbone implementation is based on
[VMamba](https://github.com/MzeroMiko/VMamba) and includes Mamba operator code.

## License

DiTrust-Net contributions are licensed under the Apache License 2.0. See
`LICENSE`. Third-party components retain their original terms; see
`THIRD_PARTY_NOTICES.md`.
