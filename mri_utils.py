
import json
from pathlib import Path
import shutil


def convert_dcm_to_niix(dcm_dir: Path, niix_dir: Path, compress: bool = False) -> None:
    """Convert DICOM files to NIfTI format using dcm2niix.

    Args:
        dcm_dir: Directory containing the DICOM files (unzipped).
        niix_dir: Directory where the converted NIfTI files will be saved.
        compress: Whether to compress the output NIfTI files. Defaults to False.


    Raises:
        FileNotFoundError: If ``dcm2niix`` is not found in the system path.
    """
    import subprocess

    # Check if dcm2niix is available
    if not shutil.which("dcm2niix"):
        raise FileNotFoundError("dcm2niix not found. Please install it and ensure it's in your PATH. Run 'uv install'")

    # Create the output directory if it doesn't exist
    niix_dir.mkdir(parents=True, exist_ok=True)

    # Run dcm2niix command
    command = ["dcm2niix", "-f", "%d_%s", "-z", "y" if compress else "n", "-o", str(niix_dir), str(dcm_dir)]
    print(f"Converting DICOM to NIfTI: {' '.join(command)}")
    subprocess.run(command, check=True)





def find_mprage(niix_dir: Path, prefer_no_correction=False) -> list[Path]:
    """Find MPRAGE anatomical scans in a directory of NIfTI conversions.

    Scans every JSON sidecar in ``niix_dir`` and selects those whose
    ``"ProtocolName"`` contains ``"MPRAGE GRAPPA2"``. By default, only
    sidecars whose ``"NonlinearGradientCorrection"`` field is truthy are
    kept; when ``prefer_no_correction`` is ``True``, only sidecars without
    nonlinear gradient correction are kept instead. For each matching
    sidecar, the corresponding ``.nii`` file is located and returned.

    Args:
        niix_dir: Directory containing the ``.json`` sidecars and their
            matching ``.nii`` files.
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
        raise FileNotFoundError(f"Directory {niix_dir} does not exist. Please convert DICOM to NIfTI first using convert_dcm_to_niix()")

    niix_json_sidecars = list(niix_dir.glob("*.json"))
    mprage_json_sidecars: list[Path] = []

    nii_files: list[Path] = []

    for json_sidecar in niix_json_sidecars:
        # convert json to dict
        with open(json_sidecar, "r") as f:
            json_dict = json.load(f)

            # Check if "ProtocolName" contains "MPRAGE GRAPPA2"
            if "ProtocolName" not in json_dict or "MPRAGE GRAPPA2" not in json_dict["ProtocolName"]:
                continue

            # Select scans based on the desired NonlinearGradientCorrection state
            has_correction = bool(json_dict.get("NonlinearGradientCorrection", False))
            if has_correction != prefer_no_correction:
                mprage_json_sidecars.append(json_sidecar)

    #  print found sidecars
    if len(mprage_json_sidecars) > 0:
        for mprage_json_sidecar in mprage_json_sidecars:
            print(f"Found MPRAGE JSON sidecar: {mprage_json_sidecar}")
            
            # Check if the corresponding NIfTI file exists
            nii_file = mprage_json_sidecar.with_suffix(".nii")

            print(f"Checking for corresponding NIfTI file: {nii_file}")

            if not nii_file.exists():
                raise FileNotFoundError(f"Corresponding NIfTI file not found for {mprage_json_sidecar}")

            nii_files.append(nii_file)

            print(f"Found MPRAGE anatomical scan: {nii_file}")

        return nii_files
    else:
        raise FileNotFoundError(f"No MPRAGE anatomical scan found in {niix_dir}")

    
def main():
    niix_dir = Path("niix")
    mprage_nii_files = find_mprage(niix_dir)
    print(f"MPRAGE NIfTI files: {mprage_nii_files}")

if __name__ == "__main__":
    main()
