"""Generate an STL from MPRAGE anatomical brain data.

Calvin Woehrle
July 1, 2026
"""

import logging
import shutil
import subprocess
import sys
from typing import cast

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import pymeshfix
import pyvista as pv
import SimpleITK as sitk
import trimesh
from nibabel.nifti1 import Nifti1Image
from scipy import ndimage

from . import brain_utils

logger = logging.getLogger(__name__)


def _select_device() -> str:
    """Pick the best available HD-BET device for the current machine.

    - macOS: always uses Apple's Metal backend (``mps``).
    - Windows / Linux: uses CUDA (``cuda``) when an NVIDIA GPU is available,
      otherwise falls back to ``cpu``.
    """
    if sys.platform == "darwin":
        return "mps"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - torch missing or CUDA probe failed
        pass
    return "cpu"


def main():

    brain_utils.setup_logging()
    brain_utils.ensure_dirs(brain_utils.BIN_DIR, brain_utils.STL_OUT_DIR, brain_utils.HTML_OUT_DIR)

    img, data, MPRAGE = brain_utils.get_img()

    # Intensity uniformity (N4 bias-field correction)
    #
    # MRI surface coils make one side of the head brighter than the other, which
    # breaks the simple percentile thresholds used below. SimpleITK's N4 estimates
    # and divides out that smooth bias field so a single threshold behaves
    # consistently across the whole volume. The corrected image is saved as a
    # `.nii.gz` and reloaded so the rest of the pipeline works in one voxel space.
    MPRAGE_N4 = brain_utils.BIN_DIR / "mprage_n4.nii.gz"

    def n4_bias_field_correct(
        in_path, out_path, shrink_factor=4, iters=(50, 50, 50, 50)
    ):
        """Estimate and remove the low-frequency intensity bias field (SimpleITK N4)."""
        img = sitk.ReadImage(str(in_path), sitk.sitkFloat32)
        mask = sitk.OtsuThreshold(img, 0, 1, 200)

        # Estimate the field on a downsampled volume for speed, then apply full-res.
        img_small = sitk.Shrink(img, [shrink_factor] * img.GetDimension())
        mask_small = sitk.Shrink(mask, [shrink_factor] * img.GetDimension())

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(list(iters))
        corrector.Execute(img_small, mask_small)

        log_bias = corrector.GetLogBiasFieldAsImage(img)
        corrected = img / sitk.Exp(log_bias)
        sitk.WriteImage(corrected, str(out_path))
        return out_path

    n4_bias_field_correct(MPRAGE, MPRAGE_N4)

    # Reload the corrected volume so all downstream steps share one voxel space.
    img_n4 = cast(Nifti1Image, nib.load(MPRAGE_N4))
    data_n4 = np.asarray(img_n4.dataobj)
    zooms = img_n4.header.get_zooms()

    logger.info(
        "N4-corrected: %s  voxel %s mm",
        img_n4.shape,
        tuple(round(float(z), 2) for z in zooms[:3]),
    )
    logger.info("Intensity:    min=%.0f  max=%.0f", data_n4.min(), data_n4.max())

    # Brain extraction with HD-BET
    #
    # The erode/dilate trick previously used to strip the skull is fragile. HD-BET
    # (MIC-DKFZ) is a neural-network brain-extraction tool that produces an
    # accurate brain mask across MRI contrasts. It runs on the N4-corrected image and
    # writes a binary mask, which we load as `brain_mask`. The first run downloads the
    # model weights.
    BRAIN_OUT = brain_utils.BIN_DIR / "mprage_brain.nii.gz"
    BRAIN_MASK = brain_utils.BIN_DIR / "mprage_brain_bet.nii.gz"  # cached HD-BET result

    def run_hd_bet(in_path, out_path, mask_path, device="cpu", force=False):
        """Run HD-BET brain extraction, caching the mask so it only runs once.

        If ``mask_path`` already exists it is reused (no computation). Set
        ``force=True`` or delete the file to recompute.
        """
        if mask_path.exists() and not force:
            logger.info("Using cached brain mask: %s", mask_path)
            return mask_path

        hd_bet = shutil.which("hd-bet")
        if hd_bet is None:
            raise RuntimeError("hd-bet CLI not found; run `uv sync` to install it.")

        cmd = [
            hd_bet,
            "-i",
            str(in_path),
            "-o",
            str(out_path),
            "-device",
            device,
            "--save_bet_mask",
        ]
        if device == "cpu" or device == "mps":
            cmd.append("--disable_tta")  # much faster on CPU, minimal quality loss
        subprocess.run(cmd, check=True)

        # HD-BET writes the mask next to the output with a "bet" suffix.
        stem = out_path.name[: -len(".nii.gz")]
        masks = list(out_path.parent.glob(f"{stem}*bet*.nii.gz"))
        if not masks:
            raise FileNotFoundError("HD-BET did not produce a mask file.")
        return masks[0]

    mask_path = run_hd_bet(MPRAGE_N4, BRAIN_OUT, BRAIN_MASK, device=_select_device())
    brain_mask = np.asarray(cast(Nifti1Image, nib.load(mask_path)).dataobj) > 0
    logger.info(
        "Brain mask: %s voxels  from %s", f"{int(brain_mask.sum()):,}", mask_path.name
    )

    # Use spacing to consider original voxel size when generating the mesh.
    spacing = tuple(float(z) for z in zooms[:3])

    def largest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component of a boolean mask."""
        labels, n = ndimage.label(mask)
        if n > 1:
            largest = np.argmax(np.bincount(labels.ravel())[1:]) + 1
            mask = labels == largest
        return mask

    def cap_base(
        mask: np.ndarray, axis: int = 2, trim: int = 3, pad: int = 2
    ) -> np.ndarray:
        """Flatten the neck opening into a solid base so the mesh is watertight.

        Cuts off the ragged bottom ``trim`` slices along the inferior-superior
        ``axis`` (Z / superior for RAS data) so the remaining voxels end on a flat
        plane, then pads the whole volume with background so every other surface
        closes during marching cubes. The result is a closed shell that stands flat.
        """
        m = mask.copy()
        present = np.where(m.any(axis=tuple(a for a in range(m.ndim) if a != axis)))[0]
        base = present.min() + trim
        sl = [slice(None)] * m.ndim
        sl[axis] = slice(0, base)
        m[tuple(sl)] = False  # remove everything below the base plane
        return np.pad(m, pad, mode="constant", constant_values=False)

    def mesh_from_mask(
        mask: np.ndarray,
        mask_sigma: float = 0.8,
        taubin_iter: int = 30,
        pass_band: float = 0.1,
        reduction: float = 0.6,
        watertight: bool = True,
    ):
        """Binary mask -> smoothed, decimated, watertight triangle surface (mm units).

        Marching cubes (via VTK ``contour``) gives a staircased surface, so we then
        apply volume-preserving **Taubin smoothing** and **quadric decimation** to
        get an organic-looking, lightweight mesh. Finally **MeshFix** removes
        non-manifold edges and fills every hole, yielding a closed manifold shell.
        """
        vol = ndimage.gaussian_filter(mask.astype(np.float32), sigma=mask_sigma)

        grid = pv.ImageData(dimensions=vol.shape, spacing=spacing)
        grid.point_data["v"] = vol.flatten(order="F")

        surf = grid.contour([0.5]).triangulate()
        surf = surf.smooth_taubin(n_iter=taubin_iter, pass_band=pass_band)
        if reduction:
            surf = surf.decimate(reduction)
        surf = surf.clean()

        verts = np.asarray(surf.points)
        faces = surf.faces.reshape(-1, 4)[:, 1:]

        if watertight:
            verts, faces = pymeshfix.clean_from_arrays(verts, faces)

        return verts, faces

    # ============ Whole Head Mesh Extraction ============
    # Threshold head vs. air (on the N4-corrected volume), fill interior holes,
    # then keep the largest connected blob.
    head_thr = np.percentile(data_n4[data_n4 > 0], 60)
    head_mask = ndimage.binary_fill_holes(data_n4 > head_thr)
    head_mask = largest_component(head_mask)

    # Close the open neck into a flat base so the head prints as a solid, watertight
    # shell. Keep the uncapped head_mask for the skull step below.
    head_mask_capped = cap_base(head_mask, axis=2, trim=3, pad=2)

    verts, faces = mesh_from_mask(head_mask_capped, mask_sigma=0.8, reduction=0.6)
    logger.info(
        "Head:  %s verts, %s tris  (thr=%.0f, capped base)",
        f"{len(verts):,}",
        f"{len(faces):,}",
        head_thr,
    )

    # ============ Brain Mesh Extraction ============
    # Use the HD-BET neural-network brain mask instead of erode/dilate heuristics.
    brain_mask = largest_component(brain_mask)
    brain_mask = ndimage.binary_fill_holes(brain_mask)

    verts_brain, faces_brain = mesh_from_mask(brain_mask, mask_sigma=1.0, reduction=0.5)
    logger.info(
        "Brain: %s verts, %s tris  (HD-BET)",
        f"{len(verts_brain):,}",
        f"{len(faces_brain):,}",
    )

    # ============ Skull Mesh Extraction ============
    # Cortical bone is dark on T1, so segment the skull geometrically: peel the
    # outer scalp off the head, then remove the (slightly padded) brain region.
    # What remains is the bony shell between scalp and brain.
    scalp_peeled = ndimage.binary_erosion(head_mask, iterations=4)  # drop outer scalp
    brain_padded = ndimage.binary_dilation(
        brain_mask, iterations=2
    )  # small safety margin
    skull_mask = scalp_peeled & ~brain_padded
    skull_mask = largest_component(skull_mask)

    verts_skull, faces_skull = mesh_from_mask(skull_mask, mask_sigma=0.8, reduction=0.6)
    logger.info(
        "Skull: %s verts, %s tris",
        f"{len(verts_skull):,}",
        f"{len(faces_skull):,}",
    )

    # Interactive 3D head model
    #
    # The same marching-cubes mesh (`verts`, `faces`) is drawn with Plotly, so you
    # can click-drag to rotate, scroll to zoom, and right-drag to pan the model in
    # real time.
    # Reuse the whole-head mesh from the marching-cubes step above.
    x, y, z = verts.T
    i, j, k = faces.T

    head = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        color="rgb(217, 191, 179)",
        opacity=1.0,
        lighting=dict(ambient=0.45, diffuse=0.9, specular=0.3, roughness=0.5),
        lightposition=dict(x=200, y=200, z=400),
        flatshading=False,
        name="head",
    )

    fig = go.Figure(data=[head])
    fig.update_layout(
        title="Calvin Woehrle - Interactive 3D head model",
        width=800,
        height=800,
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",  # keep true anatomical proportions
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Optionally save a standalone, shareable HTML file:
    fig.write_html(str(brain_utils.HTML_OUT_DIR / "head_model.html"))

    # fig.show()

    # Export to STL
    #
    # The marching-cubes mesh (`verts`, `faces`) is already a watertight triangle
    # surface, so it can be written directly to an STL file for 3D printing or
    # viewing in any CAD/mesh tool. Coordinates are in millimetres, so the print
    # comes out life-size.
    def export_stl(verts, faces, path, header="MRI model - Calvin Woehrle"):
        """Build, repair and validate a triangle mesh, then write a binary STL (mm).

        Args:
            path: Destination ``.stl`` file path (not a directory).
        """
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

        # Clean up marching-cubes artefacts for a print-ready shell.
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

        mesh.export(path)  # extension (.stl) selects the exporter
        return mesh

    # ============== Outputs ================

    # ============ Export all three meshes to STL ============
    # Reuses export_stl (trimesh). Files land in STL_OUT_DIR, mm units.
    meshes = {
        "head": (verts, faces),
        "brain": (verts_brain, faces_brain),
        "skull": (verts_skull, faces_skull),
    }

    for name, (v, f) in meshes.items():
        out_path = brain_utils.STL_OUT_DIR / f"{name}.stl"
        mesh = export_stl(v, f, path=out_path)
        logger.info(
            "Wrote %s: %s triangles, %s vertices, watertight=%s",
            out_path,
            f"{len(mesh.faces):,}",
            f"{len(mesh.vertices):,}",
            mesh.is_watertight,
        )


if __name__ == "__main__":
    main()
