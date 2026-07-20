import logging

import BrainBuilder
import convert
import mri_utils as mri

logger = logging.getLogger(__name__)


def main():
    """Entry point for the MRI project."""
    mri.setup_logging()
    logger.info("Hello Calvin! Welcome to the MRI project.")


def run_conversion():
    """Run the DICOM to NIfTI conversion process."""
    convert.main()


def run_brain_extraction():
    """Run the brain extraction process. Generate STL and HTML outputs."""
    BrainBuilder.main()


def run_3way_viewer():
    """Run the 3-way viewer for the MPRAGE anatomical scan."""
    mri.generate_3way_viewer()


if __name__ == "__main__":
    main()
