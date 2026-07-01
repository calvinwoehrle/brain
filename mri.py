"""Interpret a 4D fMRI run produced by dcm2niix.

Loads a NIfTI BOLD series and produces the standard "first looks":
  1. Basic header info (shape, voxel size, TR, number of volumes).
  2. Mean BOLD image (anatomy-like reference).
  3. Temporal standard deviation map (where the signal fluctuates = candidate activity).
  4. Temporal SNR map (mean / std) -- data quality.
  5. A single voxel's BOLD time series (the raw "activity" trace).
  6. A carpet/"grayplot" of all voxels over time (motion & artifact check).
"""

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting
from nilearn.image import mean_img

NIIX_DIR = Path("niix")


def load(path: Path) -> Nifti1Image:
    return cast(Nifti1Image, nib.load(path))


def pick_run() -> Path:
    """Pick the 4D NIfTI with the most volumes (the real task run)."""
    best, best_nt = None, -1
    for f in sorted(NIIX_DIR.glob("*.nii")):
        img = load(f)  # header only, data not loaded yet
        nt = img.shape[3] if img.ndim == 4 else 1
        if nt > best_nt:
            best, best_nt = f, nt
    if best is None:
        raise FileNotFoundError(f"No .nii files found in {NIIX_DIR}/")
    return best


def main() -> None:
    # run = pick_run()
    run = Path("niix/sub-SUBJECT_task-task.nii")
    img = load(run)
    data = img.get_fdata()  # shape (X, Y, Z, T)
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else None

    print(f"File:        {run.name}")
    print(f"Shape:       {img.shape}  (X, Y, Z, time)")
    print(f"Voxel size:  {tuple(round(z, 2) for z in zooms[:3])} mm")
    print(f"TR:          {tr} s")
    print(f"Volumes:     {img.shape[3] if img.ndim == 4 else 1}")
    print(f"Duration:    {round(img.shape[3] * tr, 1) if tr else '?'} s")

    if img.ndim != 4:
        print("Not a 4D series -- nothing temporal to interpret.")
        return

    # --- Derived maps over the time dimension ------------------------------
    mean_vol = data.mean(axis=3)
    std_vol = data.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(std_vol > 0, mean_vol / std_vol, 0)

    mean_nii = mean_img(img)
    std_nii = nib.Nifti1Image(std_vol, img.affine, img.header)
    tsnr_nii = nib.Nifti1Image(tsnr, img.affine, img.header)

    plotting.plot_epi(mean_nii, title="Mean BOLD (reference)")
    plotting.plot_stat_map(
        std_nii, bg_img=mean_nii, title="Temporal SD (fluctuation = candidate activity)",
        colorbar=True,
    )
    plotting.plot_stat_map(
        tsnr_nii, bg_img=mean_nii, title="Temporal SNR (data quality)", colorbar=True,
    )

    # --- One voxel's BOLD time series (the raw activity trace) -------------
    # Pick the brightest in-brain voxel as an example.
    cx, cy, cz = np.unravel_index(np.argmax(mean_vol), mean_vol.shape)
    ts = data[cx, cy, cz, :]
    t = np.arange(img.shape[3]) * (tr or 1)

    fig, ax = plt.subplots(figsize=(9, 3))
    # fig.suptitle(f"Voxel at ({cx}, {cy}, {cz}) -- mean signal = {mean_vol[cx, cy, cz]:.1f}")
    ax.plot(t, ts, lw=1)
    ax.set_title(f"BOLD time series at voxel ({cx}, {cy}, {cz})")
    ax.set_xlabel("Time (s)" if tr else "Volume")
    ax.set_ylabel("Signal")
    fig.tight_layout()

    # --- Carpet plot: every brain voxel over time -------------------------
    brain = mean_vol > mean_vol.mean()  # crude brain mask
    carpet = data[brain]  # (n_voxels, T)
    carpet = (carpet - carpet.mean(axis=1, keepdims=True))  # detrend mean
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.imshow(carpet, aspect="auto", cmap="gray", vmin=-50, vmax=50)
    ax2.set_title("Carpet plot (voxels x time) -- look for stripes = motion")
    ax2.set_xlabel("Volume")
    ax2.set_ylabel("Voxel")
    fig2.tight_layout()

    plotting.show()
    plt.show()


if __name__ == "__main__":
    main()
