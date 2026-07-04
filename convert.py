# Calvin Woehrle, 2026

from pathlib import Path

from mri_utils import convert_dcm_to_niix, get_user_input


def main():
    try:
        dcm_dir = get_user_input()

        niix_dir = Path("niix")

        convert_dcm_to_niix(dcm_dir, niix_dir)

        print(
            f"Successfully converted DICOM files from {dcm_dir} to NIfTI in {niix_dir}."
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
