
import json
from pathlib import Path


# Find the MPRAGE anatomical scan in the niix directory.
# The MPRAGE scan is identified by the "ProtocolName" field in the JSON sidecar file, which should contain "MPRAGE GRAPPA2", and the "NonlinearGradientCorrection" field should be set to True.
def find_mprage(niix_dir: Path) -> list[Path]:
    niix_json_sidecars = list(niix_dir.glob("*.json"))
    mprage_json_sidecars: list[Path] = []

    nii_files: list[Path] = []

    for json_sidecar in niix_json_sidecars:
        # convert json to dict
        with open(json_sidecar, "r") as f:
            json_dict = json.load(f)
            
            # Check if "ProtocolName" is "MPRAGE GRAPPA2" and "NonlinearGradientCorrection" is True
            if "ProtocolName" in json_dict and "MPRAGE GRAPPA2" in json_dict["ProtocolName"] and "NonlinearGradientCorrection" in json_dict and json_dict["NonlinearGradientCorrection"]:
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
