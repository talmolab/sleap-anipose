# sleap-anipose

[![CI](https://github.com/talmolab/sleap-anipose/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-anipose/actions/workflows/ci.yml)
[![Lint](https://github.com/talmolab/sleap-anipose/actions/workflows/lint.yml/badge.svg)](https://github.com/talmolab/sleap-anipose/actions/workflows/lint.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-anipose/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-anipose)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-anipose?label=Latest)](https://github.com/talmolab/sleap-anipose/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-anipose?label=PyPI)](https://pypi.org/project/sleap-anipose)

SLEAP to Anipose triangulation pipeline for 3D multi-animal pose tracking.

## Installation
```
pip install sleap-anipose
```

If you are using conda 
```
conda create -n sleap-anipose python=3.8
conda activate sleap-anipose
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
slap-draw_board --board_name my/path/board.jpg --board_x 8 --board_y 11 --square_length 24.0 --marker_length 18.75 --marker_bits 4 --dict_size 1000 --img_width 1440 --img_height 1440 --save my/path/board.toml
```

The draw_board function saves both an image of the board design and a board parameters file. 

The requried `board_name` parameter specifies the path to save the board to, and the file extension can end in any open-cv suitable image format such as `.jpg` and `.png`. 

The required `board_x` and `board_y` arguments specify the number of squares along the width and height of the board respectively. The `square_length` and `marker_length` arguments specify the length of the checkerboard square edges and marker square edges respectively. They can be in any units so long as they are the same. 

The required `marker_bits` and `dict_size` arguments are related to the aruco markers themselves and detail the number of bits used to make each marker and the size of the dictionary that the markers are sampled from. The `image_width` and `image_height` arguments specify the number of pixels along the image width and height respectively. 

Lastly, the `save` argument is an optional input that specifies the path to save a board parameters file to, and it must end in `.toml`. The board parameter file compiles the `board_x`, `board_y`, `square_length`, `marker_length`, `marker_bits`, `dict_size` arguments into a `.toml` file that is then used during the calibration procedure. **This file is mandatory to run calibration**. It is important to note that if the optional `save` parameter is not given, the `.toml` parameter file will not be saved using the `draw_board` function.

If a user already has a board design and only needs to generate the parameters file, the user can take advantage of the write_board function, which takes in the same types of arguments that draw_board does. 

```
slap-write_board --board_name my/path/board.toml --board_x 8 --board_y 11 --square_length 24.0 --marker_length 18.75 --marker_bits 4 --dict_size 1000
```

3. Calibrate your camera setup by displaying the calibration board to all cameras. Make sure to do this before running any experiments. See [`CALIBRATION_GUIDE.md`](docs/CALIBRATION_GUIDE.md) for more information.

4. Track your data using SLEAP. Make sure that animal tracks are matching across multiple views. See [`SLEAP_GUIDE.md`](docs/SLEAP_GUIDE.md) for more information.

5. Set up your data according to [`FOLDER_STRUCTURE.md`](docs/FOLDER_STRUCTURE.md).

6. Generate the calibration for your cameras:
```
slap-calibrate --session my/path --board my/path/board.toml --excluded_views side --excluded_views top --calib_fname my/path/calibration.toml --metadata_fname my/path/calibration_metadata.h5 --histogram_path my/path/reprojection_histogram.png --reproj_path my/path
```

The required `session` argument is a path pointing to the session folder to calibrate. This folder should contain the calibration videos and / or images within its view subfolders. 

The required `board` argument points to the calibration board parameters file that was created with either the draw_board or write_board function. 

The required `calib_fname` argument specifies the path to save the calibration results to and it must end in `.toml`. 

The optional multiple input `excluded_views` argument specifies camera views that are to be excluded from calibration. These arguments are the names for the cameras to be excluded from calibration, not the paths pointing to the view subfolders. To specify more than 1 view, the parameter must be called multiple times as shown above. 

The `metadata_fname` argument specifies the path to save the calibration metadata and it must end in .h5. This file contains some useful information about the board detections that were used for calibration, and more importantly, these board detections can later be used to assess the quality of calibration. **It is important to note that this file will not save unless the user inputs a path to save it to.**

The `histogram_path` argument points to the path to save the histogram of reprojection errors to. Lastly, the `reproj_path` argument points to the session to save the images contrasting calibration board corner detections against reprojected corner positions. **It is important to note that neither of these files will save unless the user inputs a path to save them to.**

8. Generate the triangulated 3D points based on your tracking and calibration:

```
slap-triangulate --p2d my/path --calib my/path/calibration.toml --fname p3d.h5 --frames 1000 2000 --excluded_views side --excluded_view top --fname my/path/points3d.h5 --disp_progress True \
--constraints 0 1 --constraints 2 3 --weak_constraints 4 5 --scale_smooth 3 --scale_length 4 --scale_length_weak 1 --reproj_error_threshold 10 --reproj_loss l2 --n_deriv_smooth 2
```

The required `p2d` argument points to the session containing the tracked and proofread files to be used for triangulation. 

The required `calib` argument is a path pointing to the .toml file containing the results of calibration.

The required `fname` argument is the path to save the triangulated points to and it muse end in .h5.

The optional `frames` argument is a tuple covering the range of frames to triangulate (inclusive lower end, exclusive upper end).

The multi-input optional `excluded_views` argument specifies the camera views to exclude in the triangulation process. The only restriction on this feature is that the views used for triangulation must a subset of the views found in the calibration file.  

The optional `disp_progress` argument is a flag that determines whether or not to display the progress of the triangulation procedure (default False). 

Aniposelib has the capability to optimize for spatiotemporal consistency along with accuracy in its optimization process. The user can take full advantage of this feature using the additional keyword arguments shown in the triangulation function. The optional `constraints` and `constraints_weak` arguments detail which edges in the animal skeleton have a fixed length. Whereas the `constraints` argument corresponds to rigid limbs such as arm length in human beings, the `constraints_weak` argument corresponds to more flexible limbs such as the tarsus length in flies. Both are K x 2 arrays, with K being the number of constraints. In the example shown above, `--constraints 0 1 --constraints 2 3` denotes two different rigid limbs, one between nodes 0 and 1, and the other between nodes 2 and 3. The same logic extends to the `constraints_weak` argument, where there is a weak constraint between nodes 4 and 5.

The optional `scale_smooth` argument denotes the weight of temporal smoothing in the optimization procedure, and it is set to 4 by default.

The optional `scale_length` and `scale_length_weak` arguments denote the weight of their respective constraints in the optimization loss function. They are 2 and 0.5 respectively by default.

The triangulation algorithm rejects points that have a reprojection error above the threshold (points are roughly triangulated and reprojection errors are calculated before the final triangulation step is performed), which is denoted by the optional `reproj_error_threshold` argument. This argument is set to 15.0 by default.

The user can change the loss function with the optional `reproj_loss` argument, which is 'soft_l1 by default. See [the scipy least squares documentation](scipy.optimize.least_squares) for more information regarding the loss function options.

Lastly, the optional `n_deriv_smooth` argument details the order of derivative to smooth for in the temporal filtering process, and it is 1 by default. To learn more, refer to the [aniposelib documentation](https://github.com/lambdaloop/aniposelib/tree/master/aniposelib).


**As a script**

```python
import sleap_anipose as slap

slap.draw_board(board_name = 'my/path/board.jpg', 
                board_x = 8, 
                board_y = 11, 
                square_length = 24.0, 
                marker_length = 18.75, 
                marker_bits = 4, 
                dict_size = 1000, 
                img_width = 1440, 
                img_height = 1440, 
                save = 'my/path/board.toml')

cgroup, metadata = slap.calibrate(session = "my/path", 
                                board = "my/path/board.toml", 
                                excluded_views = ('side', 'top'),
                                calib_fname = "my/path/calibration.toml", 
                                metadata_fname = "my/path/calibration.metadata.h5", 
                                histogram_path = "my/path/reprojection_histogram.png", 
                                reproj_path = "my/path")


points3d = slap.triangulate(p2d = 'my/path', 
                            calib = 'my/path/calibration.toml', 
                            frames = (1000, 2000), 
                            excluded_views = ('side', 'top'),
                            fname = 'my/path/points3d.h5', 
                            disp_progress=True, 
                            constraints = [[0,1], [2,3]], 
                            constraints_weak = [[4,5]],
                            scale_smooth = 3, 
                            scale_length = 4, 
                            scale_length_weak = 1, 
                            reproj_error_threshold = 10, 
                            reproj_loss = 'l2', 
                            n_deriv_smooth = 2)
```

*It is important to note that while saving is automatic from the command line, it is optional when running the commands in a script or notebook.*