"""Streamlined MPRAGE -> brain STL (HD-BET only, no N4 bias correction).

HD-BET normalizes each input internally and is robust to the coil intensity
bias field, so the brain mask it produces on the raw MPRAGE is effectively
identical to one produced on an N4-corrected volume. This module therefore skips
N4 entirely for a faster brain-only path. The N4-based head + skull pipeline
lives in :mod:`cwoehrle.brain.BrainBuilder`.
"""

import logging
from typing import cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage

from . import brain_utils
from .BrainBuilder import largest_component, mesh_from_mask, run_hd_bet
from .stl import export_meshes

logger = logging.getLogger(__name__)


def build_brain_stl() -> None:
    """Extract the brain from the raw MPRAGE and export a print-ready STL.

    Pipeline: load MPRAGE -> HD-BET brain mask (on the raw volume) -> keep the
    largest component and fill interior holes -> smoothed, watertight surface ->
    ``brain.stl`` (mm units) in :data:`brain_utils.STL_OUT_DIR`.
    """
    brain_utils.setup_logging()
    brain_utils.ensure_dirs(brain_utils.BIN_DIR, brain_utils.STL_OUT_DIR)

    img, _, MPRAGE = brain_utils.get_img()
    zooms = img.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))

    # Brain extraction with HD-BET directly on the raw MPRAGE (no N4 needed).
    brain_out = brain_utils.BIN_DIR / "brain.nii.gz"
    brain_mask_cache = brain_utils.BIN_DIR / "brain_bet.nii.gz"  # cached HD-BET result
    mask_path = run_hd_bet(
        MPRAGE, brain_out, brain_mask_cache, device=brain_utils.select_device()
    )
    brain_mask = np.asarray(cast(Nifti1Image, nib.load(mask_path)).dataobj) > 0
    logger.info(
        "Brain mask: %s voxels  from %s", f"{int(brain_mask.sum()):,}", mask_path.name
    )

    brain_mask = largest_component(brain_mask)
    brain_mask = ndimage.binary_fill_holes(brain_mask)

    verts, faces = mesh_from_mask(brain_mask, spacing, mask_sigma=1.0, reduction=0.5)
    logger.info(
        "Brain: %s verts, %s tris  (HD-BET, no N4)",
        f"{len(verts):,}",
        f"{len(faces):,}",
    )

    export_meshes({"brain": (verts, faces)}, brain_utils.STL_OUT_DIR)


def main() -> None:
    build_brain_stl()
    logger.info("Your brain STL is in %s", brain_utils.STL_OUT_DIR)


if __name__ == "__main__":
    main()
