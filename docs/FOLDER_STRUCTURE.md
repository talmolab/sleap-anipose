# Folder structure

There are 3 pre-requisites that must be met to use the sleap-anipose package.

1. **All tracking data must be proofread with [SLEAP](https://sleap.ai) and exported to an analysis.h5 file.**

2. **The user has completed taking calibration videos or images.**

3. **User data is organized according to the notion of a *session***.

A session is a folder that contians videos for a single continuous recording, i.e. **there is only 1 video per camera view in a session folder.** Sessions have a unique internal structure to keep track of files which are specific to the entire recording and which files are specific to the camera view. The basic structure of a session is shown below, with pre-requisite files and folders ending in an asterisk. 

```
session_0
├── calibration.toml
|   calibration_metadata.h5
|   reprojection_errors.png
|   points3d.h5
|   back/* (View Directories)
|   mid/* 
|   side/*
└── top/*
    ├── top_view_recording.mp4*
    |   top_view_tracks.slp
    |   top_view_tracks.proofread.analysis.h5*
    |   reprojected_points2d.h5
    |   reprojection_0.png
    |   reprojection_19.png
    |   reprojection_77.png
    |   reprojection_124.png
    └── calibration_images/*
        ├── top_calibration_video.MOV*
        |   top_view_frame_0.jpg*
        |   top_view_frame_1.jpg*
        |           .
        |           .
        |           .
        └── top_view_frame_137.jpg*
```

## Dissecting session structure

Within each session folder there are a series of subfolders corresponding to each camera view. In the diagram above, alongside these view subfolders there is a collection of files derived from running functions from sleap-anipose. `calibration.toml`, `calibration.metadata.h5`, and `reprojection_errors.png` can be created from running the [calibration function and its relevant ancilliary function](sleap_anipose/calibration.py) while 'points3d.h5' is created from running the [triangulation function](sleap_anpipose/triangulation.py). In addition, while these files are saved to the session directory, it is not absolutely necessary to save them there. However, it is highly recommended to save these files to their respective sessions. 

On the other hand, the view subfolders are absolutely mandatory and must exist in the session before running calibration or triangulation in sleap-anipose. Furthermore, within each view subfolder it is mandatory to have the following:

1. **The video of the experiment taken from the corresponding camera view**

2. **Your tracked and proofread data saved to 2 files, a .slp file and an analysis.h5 file**

3. **A subfolder named calibration_images containing either calibration board images or videos for the session and view.**

When it comes to the tracking data, the .slp file will correspond to the labeling project itself while the analysis.h5 will correspond to the finalized proofread output of the project. While you don't need to save the project file to the same folder as the analysis file, it is highly recommended to do so. You can generate the `analysis.h5` file in SLEAP by exporting your project to a .h5 file and saving it to the appropriate folder (see the [tracking guide](sleap_anipose/docs/SLEAP_GUIDE.md) for more information). It follows that when exporting the labels to a .h5 file, one should keep the .analysis.h5 extension appended to the filename of choice. In the calibration images subfolder you can put either videos or images of your calibration board. If there is no video, sleap-anipose will generate the video itself from the given images and save it to the calibration images subfolder. 

The other files found in the view subfolder are a result of running the triangulation and calibration functions within the sleap-anipose package. The `reprojected_points2d.h5` file is derived from running the [reproject function](sleap_anipose/triangulation.py) and it contains the view specific reprojected points along with the depth of the points from the root joint of the skeleton. The user has the option to save the file to another location, but it is highly recommended to save them to the appropriate session view. The `reprojection_0.png`, ..., `reprojection_124.png` files are images generated from running the [calibration function and its relevant ancilliary functions](sleap_anipose/calibration.py). The user can specify the session to save these files to, but they will always be saved as `reprojection-frame.png` within the appropriate view subfolder for that session. 

As long as the data is organized in a series of session folders one can take full advantage of sleap-anipose. 

## Benchmark Dataset Structure 

If you are using the benchmark dataset, the structure follows the same session pattern detailed earlier and these sessions are organized into parent date folders. This is shown below. 

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
        ├── calibration.toml
        |   calibration_metadata.h5 
        |   points3d.h5
        |   reprojection_errors.png
        |   back/ (View Directories)
        |   mid/
        |   side/
        └── top/
            ├── top-04182022182439-0000_h265_CRF14_denoised.mp4
            |   top-04182022182439.predictions.slp
            |   top-04182022182439.predictions.proofread.slp
            |   reprojected_points2d.h5
            |   reprojection-0.png
            |   reprojection-19.png
            |   reprojection-77.png
            |   reprojection-124.png 
            └── calibration_images/
                ├── 04182022182439-top-calibration.MOV*
                |   top-04182022182439-0.jpg
                |   top-04182022182439-1.jpg
                |           .
                |           .
                |           .
                └── top-04182022182439-137.jpg
```

Certain files and folders follow a certain naming pattern found all throughout the benchmark dataset. Some naming conventions are detailed below.

- Session directories in the benchmark dataset are named for the date they are recorded on and then the time within the day the recording started.

    For example, session `04182022182439` can be deduced as having started on 04-18-2022 at 18 hours, 24 minutes, 39 seconds. 

- The video found in each view folder is named following the convention: `{view}-{session}-{suffix}.mp4`. The '0000_h265_CRF14_denoised' suffix found in the benchmark dataset is a result of the video rendering software.

- The slp files are named following the convention: `{view}-{session}.predictions.{prediction_suffix}.slp`. If the file contains the proofread tracks, the .proofread suffix is added to the file name. 

- The images of the reprojected corners and detected corners overlaid on the calibration board are named according to the convention: `reprojection-{frame_number}.png`. The frame number is derived from the 
frame of the calibration board used. 

- The rendered movies of the calibration images are named following the convention: `{view}-{session}-calibration.MOV`.

- The calibration images are labeled according to the convention: `{view}-{session}-{frame_number}.jpg`. 
