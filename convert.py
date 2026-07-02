# Calvin Woehrle, 2026

from pathlib import Path
from tkinter import Tk, filedialog

from mri_utils import convert_dcm_to_niix

def get_user_input() -> Path:
    """Prompt the user to pick the DICOM input directory.

    Opens a native folder-selection dialog using tkinter. The NIfTI output
    directory is assumed to be ``niix`` in the current working directory.

    Returns:
        A tuple of ``(dcm_dir, niix_dir)`` as :class:`~pathlib.Path` objects.

    Raises:
        ValueError: If the user cancels the dialog without selecting a folder.
    """
    # Hide the empty root window that tkinter creates.
    root = Tk()
    root.withdraw()

    dcm_dir = filedialog.askdirectory(title="Select the DICOM directory", initialdir=Path.home() / "Downloads")
    if not dcm_dir:
        root.destroy()
        raise ValueError("No DICOM directory selected.")
    print(f"Selected DICOM directory: {dcm_dir}")

    root.destroy()

    return Path(dcm_dir)


def main():
    try:
        dcm_dir = get_user_input()

        niix_dir = Path("niix")

        convert_dcm_to_niix(dcm_dir, niix_dir)
        
        print(f"Successfully converted DICOM files from {dcm_dir} to NIfTI in {niix_dir}.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()