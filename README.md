# sleap-anipose

[![CI](https://github.com/talmolab/sleap-anipose/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-anipose/actions/workflows/ci.yml)
[![Lint](https://github.com/talmolab/sleap-anipose/actions/workflows/lint.yml/badge.svg)](https://github.com/talmolab/sleap-anipose/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-anipose/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-anipose)

SLEAP to Anipose triangulation pipeline for 3D multi-animal pose tracking.

## Installation
```
pip install sleap-anipose
```

### Development
For development, use the following syntax to install in editable mode:
```
conda env create -f environment.yml
```
This will create a conda environment called `sleap-anipose`.

To run tests, first activate the environment:
```
conda activate sleap-anipose
```
Then run `pytest` with:
```
pytest tests
```
To start fresh, just delete the environment:
```
conda env remove -n sleap-anipose
```

## Usage

**From the command line**

1. Activate the conda environment (assuming you installed via conda).
```
conda activate sleap-anipose
```

2. Generate a calibration board matching the details of the board presented before the cameras. 

```
slap-write_board --fname board.toml --board_width 8 --board_height 11 --square_length 24.0 --marker_length 18.5 --marker_bits 4 dict_size 1000
```

3. If you do not have a calibration board already, you can have the calibration board drawn and saved to a pdf file from the file made in step 2. 

```
slap-draw_board --fname board.pdf --board board.toml 
```

4. Calibrate your camera setup by displaying the calibration board to all cameras. Make sure to do this before running any experiments. See [`CALIBRATION_GUIDE.md`](docs/CALIBRATION_GUIDE.md) for more information.

5. Track your data using SLEAP. Make sure that animal tracks are matching across multiple views. See [`SLEAP_GUIDE.md`](docs/SLEAP_GUIDE.md) for more information.

6. Set up your data according to [`FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md).

7. Generate the calibration for your cameras:
```
slap-calibrate --session date_directory/session_directory --board board.toml --save_calib True --save_folder date_directory/session_directory --save_metadata True --histogram True -reproj_imgs True --save_hist True --save_reproj_imgs True
```

8. Generate the triangulated 3D points based on your tracking and calibration:
```
slap-triangulate --p2d date_directory/session_directory --calib date_directory/session_directory/calibration.toml --save True --session date_directory/session_directory --disp_progress False
```

**As a script**

```python
import sleap_anipose as slap

session = "path/to/data"

slap.calibrate(session, *calibration_args) 
slap.triangulate(session, *triangulation_args)
```

# TODO: unpack function args and explain them. 

See [`FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md) for details on how session data should be organized.