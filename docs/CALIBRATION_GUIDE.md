# Camera Calibration 

The purpose of camera calibration is to gain a quantitative understanding of how the cameras relate to each other in space and how they differ from each other due to their internal mechanisms. 
To model the spatial relationships of cameras and their internal specifications, we rely on 2 sets of matrices for each camera, the extrinsics and intrinsics matrices respectively. Once we have these sets of matrices, we can then triangulate 3D poses from our tracked 2D poses. 

# Practical Considerations

## Calibration Board 

A calibration board is a rectangular surface with a recognizable pattern that allows for quick detection via modern computer vision software. The most commonly used types of boards are checkerboards, aruco boards (which are rectangular grids with unique markers), and charuco boards (which are a combination of the two). 

**INSERT IMAGE HERE**

Calibration boards are commonly made by pasting a printed board design onto a rigid surface, however, they can also be produced via laser cutting into acryllic or other methods. Regardless of the method, it is paramount that the board design is clearly visible to the cameras. 

One can create a charuco board pattern via two different ways using sleap-anipose.

1. Through the CLI

```
slap-write_board --board_name board --img_format jpg --board_X 8 --board_Y 11 --square_length 24.0 --marker_length --18.75
```

2. Through the API

```
python 

import sleap-anipose as slap 
slap.calibration.write_board("path/board", 'jpg', 8, 11, 24.0, 18.75)
```

Currently, we only support charuco board calibration, but plan to expand to checkerboards and aruco boards. It is also important to note that we currently only generate aruco markers with 4 bits from a size 1000 dictionary. However, if one used a calibration board with a different aruco encoding the board can still be used for calibration and triangulation using sleap-anipose. One could do so by writing a toml file that describes the board according to the attributes in the write_board function. Refer to the [`API`](sleap_anipose/calibration.py) for more details. 

## Camera Setup  

In order to get the best calibration and triangulation possible, it is useful to design the camera setup to avoid certain pitfalls. 

1. Avoid having the cameras obfuscate each other. 

**INCLUDE IMAGE HERE**

2. Avoid placing the cameras in positions that lead to animal easily occluding one another. 

**INCLUDE IMAGE HERE**

3. Avoid placing the cameras in positions that fail to capture the full 3D features of the animal.

**INCLUDE IMAGE HERE**

4. Make sure that there is enough space alotted in the experimental space for the cameras to capture videos of the calibration board at various angles and at a reasonable resolution. 

**INCLUDE IMAGE HERE**

Last but not least, it is vital to have the cameras synchronized when recording. This can be achieved with a hardware controller (such as a Rasberry Pi) and a pulse emitter. The synchronized videos must then be saved by an appropriate acquisition software and compressed if necessary. For the benchmark dataset, the camera setup consisted of 5 (4 for some sessions) FLIR Blackfly S Mono cameras, each with ThorLabs fixed focal length 3.5 or 4.5 mm lenses (MVL4WA/ 5WA). Cameras were triggered with a strobe pulse from a tdt rx8. Data was acquired with spinview software and then compressed with ffmpeg.

## Calibration Workflow 

Calibration should be carried out each time before the start of a recording session. If one is recording multiple sessions within the same day, it is fine to calibrate only at the beginning of the first session, given that the camera setup is not altered between sessions. If the camera setup is altered between sessions, calibration must be carried out once again. The workflow is usually conducted as follows.

1. Check camera synchronization and setup stability. The cameras should not move throughout the rest of the workflow and during the subsequent experiments. 

2. Place the calibration board in front of the camera setup such that it is clearly visible to as many cameras as possible. 

3. Start recording. 

4. Move the calibration board around the setup to expose it to other cameras from which it may have been occluded. Even if the board is visible to all cameras one should still move the board around so that marker detection can be robust to variability in board position. Also, it is critical to move the board around at a pace that is suitable for the framerate of the cameras, as blurry frames lead to poor detection. 

5. Stop recording. 

6. Check the videos for proper synchronization and visibility, troubleshoot if necessary. 

Alternatively, instead of taking videos for board calibration, one could take synchronized images from each view and then use sleap-anipose to stitch the frames into a movie. Refer to the [`API`](sleap_anipose/calibration.py) for more details. 