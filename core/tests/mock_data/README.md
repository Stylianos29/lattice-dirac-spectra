# Mock gauge data

This directory holds small U(1) HMC gauge ensembles used by the M3 unit tests.
The `.h5` files are **not committed** (see `.gitignore`); generate them locally with:

```bash
conda activate KLenvDev          # env with mpi4py + parallel h5py
cd ~/u1-hmc
python u1-hmc.py \
  --beta 1.0 --length 8 --n-traj 32 --n-tau 8 --traj-length 1 \
  --print-every 16 --start hot \
  --output-fname ~/lattice-dirac-spectra/core/tests/mock_data/u1_L8_beta1.000.h5
```

Expected output: `shape: (8, 8, 2) | dtype: complex128 | |U| mean: 1.0`