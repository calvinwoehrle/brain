import argparse
import logging

from . import BrainBuilder, convert
from . import brain_utils as mri

logger = logging.getLogger(__name__)


def run_conversion(args: argparse.Namespace) -> None:
    """Run the DICOM to NIfTI conversion process."""
    dcm_dir = convert.resolve_dcm_dir(args.folder)
    mri.clear_dir(mri.NIIX_DIR)
    mri.convert_dcm_to_niix(dcm_dir, mri.NIIX_DIR)
    logger.info(
        "Successfully converted DICOM files from %s to NIfTI in %s.",
        dcm_dir,
        mri.NIIX_DIR,
    )


def run_brain_extraction(args: argparse.Namespace) -> None:
    """Run the brain extraction process. Generate STL and HTML outputs."""
    BrainBuilder.main()


def run_3way_viewer(args: argparse.Namespace) -> None:
    """Run the 3-way viewer for the MPRAGE anatomical scan."""
    mri.generate_3way_viewer()


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the MRI project."""
    parser = argparse.ArgumentParser(
        prog="brain",
        description=(
            "MRI processing and 3D brain mesh visualization from DICOM/NIfTI scans."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert DICOM files to NIfTI."
    )
    convert_parser.add_argument(
        "folder",
        help="DICOM input directory.",
    )
    convert_parser.set_defaults(func=run_conversion)

    brain_parser = subparsers.add_parser(
        "brain",
        help="Run brain extraction and generate STL and HTML outputs.",
    )
    brain_parser.set_defaults(func=run_brain_extraction)

    viewer_parser = subparsers.add_parser(
        "3way",
        help="Open the 3-way viewer for the MPRAGE anatomical scan.",
    )
    viewer_parser.set_defaults(func=run_3way_viewer)

    return parser


def main():
    """Entry point for the MRI project."""
    mri.setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
