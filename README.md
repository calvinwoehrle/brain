# Brain

This is a personal project to visualize my brain. I love data and it's fun to know for sure my brain isn't smooth.

The most useful output of this library is a print-ready STL of your brain. The STL is generated from my MPRAGE anatomical scan using HD-BET for brain extraction and marching cubes for mesh generation.

## Install

Make sure you have UV and Python installed. See [UV](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

Create a new folder for this project and navigate to it in your terminal

Navigate to the GitHub release page and download the latest release of cwoehrle_brain wheel file from https://github.com/calvinwoehrle/brain/releases

Add the `cwoehrle_brain-x.x.x-py3-none-any.whl` file to the project folder.

Run the following commands to create a virtual environment, activate it, and install the wheel file:

```shell
uv venv
```

```shell
source .venv/bin/activate
```

```shell
uv pip install cwoehrle_brain-x.x.x-py3-none-any.whl
```


```shell
brain convert <PATH_TO_DICOM_FOLDER>
```

```shell
brain brain
```

After everything is finished, you can find the STL file in `output/STL/brain.stl`


## Usage

Everything is exposed through a single `brain` command with subcommands:

```shell
brain <command>
```

| Command | Description |
| --- | --- |
| `convert <folder>` | Convert a DICOM directory to NIfTI. |
| `brain` | Streamlined brain-only STL from the raw MPRAGE (HD-BET, no N4). |
| `full` | Full N4 pipeline: head + brain + skull STLs and interactive HTML. |
| `3way` | Open a 3-plane slice viewer for the MPRAGE anatomical scan. |

## Convert from DICOM to NIfTI

Unzip the DICOM files and place them in a folder, then pass that folder to the
`convert` command:

```shell
uv run brain convert /path/to/dicom
```

The converted `.nii` files and their `.json` sidecars are written to the `niix/`
folder.

## Processing

### Brain-only STL (`brain`)

Extract just the brain from the raw MPRAGE and export it as a print-ready STL:

```shell
uv run brain brain
```

This path runs **HD-BET** brain extraction directly on the raw MPRAGE. HD-BET
normalizes its input internally and is robust to coil intensity bias, so N4
correction is skipped for a faster brain-only mesh. The first run downloads the
model weights, and the mask is cached in `bin/` so subsequent runs are fast.

Output: `output/STL/brain.stl` (millimetre units).

### Full pipeline: head + brain + skull (`full`)

Extract head, brain, and skull surfaces from the MPRAGE scan and export them as
STL files:

```shell
uv run brain full
```

This pipeline:

1. Runs **N4 bias-field correction** (SimpleITK) to even out coil-related
   intensity shading. The corrected volume is cached in `bin/`.
2. Runs **HD-BET** brain extraction to produce an accurate brain mask (also
   cached in `bin/`).
3. Segments the head (threshold vs. air), brain (HD-BET mask), and skull (bony
   shell between scalp and brain).
4. Generates smoothed, watertight meshes with marching cubes, Taubin smoothing,
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
uv run brain 3way
```

The viewer is saved to `output/HTML/mprage_viewer.html` and opened automatically.

## Logging

All scripts use Python's `logging` module. Progress messages are printed at
`INFO` level by default; the level can be adjusted via
`cwoehrle.brain.brain_utils.setup_logging()`.