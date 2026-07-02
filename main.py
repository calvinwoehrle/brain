import convert
import BrainBuilder


def main():
    """Entry point for the MRI project."""
    print("Hello Calvin! Welcome to the MRI project.")


def run_conversion():
    """Run the DICOM to NIfTI conversion process."""
    convert.main()

def run_brain_extraction():
    """Run the brain extraction process. Generate STL and HTML outputs."""
    BrainBuilder.main()

if __name__ == "__main__":
    main()