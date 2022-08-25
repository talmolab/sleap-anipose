# Folder structure

This package expects that your data has already been pose tracked and proofread with
[SLEAP](https://sleap.ai), and organized according to the specifications below.

It is very important to note that any file ending in an asterisk can be generated from running certain functions in the triangulation and calibration modules. 
**Files and folders not ending in an asterisk must exist before running any functions from sleap-anipose, and must exist in the structure detailed below.** 

```
Project Root Directory
├── 02-24-2022/ (Date Directories)
|        .
|        .
|        .
└── 04-18-2022/
    ├── 04182022173246/ (Session Directories)
    |   04182022174515/
    |        .
    |        .
    |        .
    └── 04182022182439/
        ├── calibration.toml*
        |   calibration_metadata.h5* 
        |   points3d.h5*
        |   reprojection_errors.png*
        |   back/ (View Directories)
        |   mid/
        |   side/
        └── top/
            ├── top-04182022182439-0000_h265_CRF14_denoised.mp4
            |   top-04182022182439.predictions.slp
            |   top-04182022182439.predictions.proofread.slp
            |   reprojected_points2d.h5*
            |   board_reprojection-0.png*
            |   board_reprojection-19.png*
            |   board_reprojection-77.png*
            |   board_reprojection-124.png*   
            └── calibration_images/
                ├── 04182022182439-top-calibration.MOV*
                |   top-04182022182439-0.jpg
                |   top-04182022182439-1.jpg
                |           .
                |           .
                |           .
                └── top-04182022182439-137.jpg
```

Certain files and folders in the example shown above follow a certain naming pattern found all throughout the benchmark dataset. Some naming conventions are detailed below.

- Session directories in the benchmark dataset are named for the date they are recorded on and then the time within the day the recording started.

    For example, session 04182022182439 can be deduced as having started on 04-18-2022 at 18 hours, 24 minutes, 39 seconds. 

- View directories are named are the camera views they represent. 

- The video found in each view folder is named following the convention: `{view}-{session}-{suffix}.mp4`. The '0000_h265_CRF14_denoised' suffix found in the benchmark dataset is a result of the video rendering software.

- The slp files are named following the convention: `{view}-{session}.predictions.{prediction_suffix}.slp`. If the file contains the proofread tracks, the .proofread suffix is added to the file name. 

- The images of the reprojected corners and detected corners overlaid on the calibration board are named according to the convention: `board_reprojection-{frame_number}.png`. The frame number is derived from the 
frame of the calibration board used. 

- The rendered movies of the calibration images are named following the convention: `{view}-{session}-calibration.MOV`.

- The calibration images are labeled according to the convention: `{view}-{session}-{frame_number}.jpg`. 
