# Calvin Woehrle, 2026

import logging
from pathlib import Path

from mri_utils import convert_dcm_to_niix, get_user_input, setup_logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    dcm_dir = get_user_input()

    niix_dir = Path("niix")

    convert_dcm_to_niix(dcm_dir, niix_dir)

    logger.info(
        "Successfully converted DICOM files from %s to NIfTI in %s.", dcm_dir, niix_dir
    )


if __name__ == "__main__":
    main()
