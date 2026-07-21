# Calvin Woehrle, 2026

import argparse
import logging
from pathlib import Path

from .brain_utils import NIIX_DIR, convert_dcm_to_niix, setup_logging

logger = logging.getLogger(__name__)


def resolve_dcm_dir(folder: str) -> Path:
    """Resolve and validate the DICOM input directory from a CLI argument.

    Args:
        folder: Path to the DICOM input directory.

    Returns:
        The resolved directory as a :class:`~pathlib.Path`.

    Raises:
        NotADirectoryError: If ``folder`` does not point to an existing directory.
    """
    dcm_dir = Path(folder).expanduser()
    if not dcm_dir.is_dir():
        raise NotADirectoryError(f"DICOM input directory not found: {dcm_dir}")
    return dcm_dir


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Convert DICOM files to NIfTI.")
    parser.add_argument(
        "folder",
        help="DICOM input directory.",
    )
    args = parser.parse_args()

    dcm_dir = resolve_dcm_dir(args.folder)

    convert_dcm_to_niix(dcm_dir, NIIX_DIR)

    logger.info(
        "Successfully converted DICOM files from %s to NIfTI in %s.", dcm_dir, NIIX_DIR
    )


if __name__ == "__main__":
    main()
