import json
import logging
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from nibabel.nifti1 import Nifti1Image
from nilearn import plotting

logger = logging.getLogger(__name__)

# Default directories
NIIX_DIR = Path("niix")

BIN_DIR = Path("bin")

OUTPUT_DIR = Path("output")
STL_OUT_DIR = OUTPUT_DIR / "STL"
HTML_OUT_DIR = OUTPUT_DIR / "HTML"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a simple, timestamped format.

    Safe to call multiple times; only configures handlers once.

    Args:
        level: The minimum logging level to emit. Defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs(*dirs: Path) -> None:
    """Create the given directories (and parents) if they don't already exist."""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def clear_dir(directory: Path, confirm: bool = True) -> None:
    """Delete all contents of a directory, optionally after user confirmation.

    Prompts the user before deleting anything when ``confirm`` is ``True``. If
    the directory doesn't exist or is already empty, this is a no-op.

    Args:
        directory: The directory whose contents should be removed.
        confirm: Whether to ask the user for permission before deleting.
            Defaults to ``True``.
    """
    if not directory.exists():
        return

    entries = list(directory.iterdir())
    if not entries:
        return

    if confirm:
        response = (
            input(f"Clear {len(entries)} existing item(s) from {directory}? [y/N] ")
            .strip()
            .lower()
        )
        if response not in ("y", "yes"):
            logger.info("Keeping existing contents of %s.", directory)
            return

    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    logger.info("Cleared contents of %s.", directory)


def convert_dcm_to_niix(dcm_dir: Path, niix_dir: Path, compress: bool = False) -> None:
    """Convert DICOM files to NIfTI format using dcm2niix.

    Args:
        dcm_dir: Directory containing the DICOM files (unzipped).
        niix_dir: Directory where the converted NIfTI files will be saved.
        compress: Whether to compress the output NIfTI files. Defaults to False.


    Raises:
        FileNotFoundError: If ``dcm2niix`` is not found in the system path.
    """
    # Check if dcm2niix is available
    if not shutil.which("dcm2niix"):
        raise FileNotFoundError(
            "dcm2niix not found. Please install it and ensure it's in your PATH. Run 'uv install'"
        )

    # Create the output directory if it doesn't exist
    niix_dir.mkdir(parents=True, exist_ok=True)

    # Run dcm2niix command
    command = [
        "dcm2niix",
        "-f",
        "%d_%s",
        "-z",
        "y" if compress else "n",
        "-o",
        str(niix_dir),
        str(dcm_dir),
    ]
    logger.info("Converting DICOM to NIfTI: %s", " ".join(command))
    subprocess.run(command, check=True)


def find_mprage(niix_dir=NIIX_DIR, prefer_no_correction=False) -> list[Path]:
    """Find MPRAGE anatomical scans in a directory of NIfTI conversions.

    Scans every JSON sidecar in ``niix_dir`` and selects those whose
    ``"ProtocolName"`` contains ``"MPRAGE GRAPPA2"``. By default, only
    sidecars whose ``"NonlinearGradientCorrection"`` field is truthy are
    kept; when ``prefer_no_correction`` is ``True``, only sidecars without
    nonlinear gradient correction are kept instead. For each matching
    sidecar, the corresponding ``.nii`` file is located and returned.

    Args:
        niix_dir: Directory containing the ``.json`` sidecars and their
            matching ``.nii`` files. Defaults to ``niix`` in the current working directory.
        prefer_no_correction: If ``True``, select MPRAGE scans that do not
            have nonlinear gradient correction applied instead of those that
            do. Defaults to ``False``.

    Returns:
        A list of paths to the MPRAGE ``.nii`` files.

    Raises:
        FileNotFoundError: If ``niix_dir`` does not exist. (See ``convert_dcm_to_niix()``.)
        FileNotFoundError: If no MPRAGE sidecar is found in ``niix_dir``, or
            if a matching sidecar has no corresponding ``.nii`` file.
    """

    # check if the niix_dir exists
    if not niix_dir.exists():
        raise FileNotFoundError(
            f"Directory {niix_dir} does not exist. Please convert DICOM to NIfTI first using convert_dcm_to_niix()"
        )

    niix_json_sidecars = list(niix_dir.glob("*.json"))
    mprage_json_sidecars: list[Path] = []

    nii_files: list[Path] = []

    for json_sidecar in niix_json_sidecars:
        # convert json to dict
        with open(json_sidecar) as f:
            json_dict = json.load(f)

            # Check if "ProtocolName" contains "MPRAGE GRAPPA2"
            if (
                "ProtocolName" not in json_dict
                or "MPRAGE GRAPPA2" not in json_dict["ProtocolName"]
            ):
                continue

            # Select scans based on the desired NonlinearGradientCorrection state
            has_correction = bool(json_dict.get("NonlinearGradientCorrection", False))
            if has_correction != prefer_no_correction:
                mprage_json_sidecars.append(json_sidecar)

    #  print found sidecars
    if len(mprage_json_sidecars) > 0:
        for mprage_json_sidecar in mprage_json_sidecars:
            logger.info("Found MPRAGE JSON sidecar: %s", mprage_json_sidecar)

            # Check if the corresponding NIfTI file exists
            nii_file = mprage_json_sidecar.with_suffix(".nii")

            logger.info("Checking for corresponding NIfTI file: %s", nii_file)

            if not nii_file.exists():
                raise FileNotFoundError(
                    f"Corresponding NIfTI file not found for {mprage_json_sidecar}"
                )

            nii_files.append(nii_file)

            logger.info("Found MPRAGE anatomical scan: %s", nii_file)

        return nii_files
    else:
        raise FileNotFoundError(f"No MPRAGE anatomical scan found in {niix_dir}")


def get_img(index: int = 0) -> tuple[Nifti1Image, np.ndarray, Path]:
    """Load an MPRAGE NIfTI file and return the image, its data array, and path.

    Args:
        index: Which MPRAGE scan to load when more than one is found.
            Defaults to the first (``0``).

    Raises:
        IndexError: If ``index`` is out of range for the MPRAGE scans found.
    """
    # Find the MPRAGE anatomical scan(s) in the niix directory.
    mprage_nii_files = find_mprage()

    if len(mprage_nii_files) > 1:
        logger.warning(
            "Found %d MPRAGE scans; using index %d (%s). "
            "Pass a different index to select another.",
            len(mprage_nii_files),
            index,
            mprage_nii_files[index].name,
        )

    MPRAGE = mprage_nii_files[index]

    img = cast(Nifti1Image, nib.load(MPRAGE))

    data = np.asarray(img.dataobj)  # shape (X, Y, Z) -- a single 3D volume

    zooms = img.header.get_zooms()

    logger.info("File:        %s", MPRAGE.name)
    logger.info("Shape:       %s  (X, Y, Z)", img.shape)
    logger.info("Voxel size:  %s mm", tuple(round(z, 2) for z in zooms[:3]))
    logger.info("Orientation: %s", "".join(nib.aff2axcodes(img.affine)))
    logger.info("Intensity:   min=%.0f  max=%.0f", data.min(), data.max())

    return img, data, MPRAGE


def generate_3way_viewer() -> None:
    """Generate an interactive 3-way viewer for the MPRAGE anatomical scan.
    Saves a standalone HTML file in the output directory and opens in default browser.
    """

    img, data, MPRAGE = get_img()

    # Robust intensity window so faint tissue isn't washed out by a few bright voxels.
    vmin, vmax = np.percentile(data[data > 0], [2, 98])

    # Draggable ortho viewer of the T1 volume. Use bg_img=None so the MPRAGE itself
    # is the background, and match the robust intensity window from earlier.
    html_view = plotting.view_img(
        img,
        bg_img=None,
        cmap="gray",
        symmetric_cmap=False,
        vmin=vmin,
        vmax=vmax,
        black_bg=True,
        title="Calvin Woehrle - MPRAGE interactive viewer",
    )

    # Optionally save a standalone HTML file you can open in any browser:
    ensure_dirs(HTML_OUT_DIR)
    output_path = HTML_OUT_DIR / "mprage_viewer.html"
    html_view.save_as_html(output_path)

    # Open in the default browser using an absolute file:// URI so the browser
    # can resolve the path regardless of the current working directory.
    webbrowser.open_new_tab(output_path.resolve().as_uri())


def select_device() -> str:
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
    setup_logging()
    niix_dir = Path("niix")
    mprage_nii_files = find_mprage(niix_dir)
    logger.info("MPRAGE NIfTI files: %s", mprage_nii_files)


if __name__ == "__main__":
    main()
