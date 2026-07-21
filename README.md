# Calvin Woehrle - Brain

This is a personal project visualize my brain.  I love data and it's fun to know for sure my brain isn't smooth

## Install

This project uses UV with python 3.14

Make sure you have UV and python installed.  See https://docs.astral.sh/uv/getting-started/installation/ for instructions on how to install UV.

After, clone this repository and navigate to the project directory and run `uv sync` to install the dependencies.


## Convert from dcm to Niix

Unzip the DICOM files and place them in a folder.  Then run the following command to convert the DICOM files to NIfTI format:

```shell
uv run convert
```

A native folder picker opens so you can select the unzipped DICOM directory. The
converted `.nii` files and their `.json` sidecars are written to the `niix/`
folder.

## Processing

### Build the 3D meshes (`brain`)

Extract head, brain, and skull surfaces from the MPRAGE scan and export them as
STL files:

```shell
uv run brain
```

This pipeline:

1. Runs **N4 bias-field correction** (SimpleITK) to even out coil-related
   intensity shading.
2. Runs **HD-BET** brain extraction to produce an accurate brain mask. The first
   run downloads the model weights, and the mask is cached in `bin/` so
   subsequent runs are fast.
3. Generates smoothed, watertight meshes with marching cubes, Taubin smoothing,
   quadric decimation, and MeshFix repair.

Outputs:

- `output/STL/head.stl`, `output/STL/brain.stl`, `output/STL/skull.stl` —
  print-ready meshes in millimetre units.
- `output/HTML/head_model.html` — an interactive Plotly render of the head mesh.
- Intermediate NIfTI files (N4-corrected volume, HD-BET mask) in `bin/`.

### Interactive slice viewer (`3way`)

Open a draggable three-plane (axial / coronal / sagittal) viewer of the MPRAGE
volume in your browser:

```shell
uv run 3way
```

The viewer is saved to `output/HTML/mprage_viewer.html` and opened automatically.

## Logging

All scripts use Python's `logging` module. Progress messages are printed at
`INFO` level by default; the level can be adjusted via
`cwoehrle.brain.mri_utils.setup_logging()`.





uv run convert<PATH_TO_DICOM_FOLDER>