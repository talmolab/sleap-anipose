"""This module defines utilities related to calibration."""

import numpy as np
import matplotlib.pyplot as plt
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames


def make_histogram(session, reprojection_error):
    """Make a visualization histogram of the reprojection errors."""
    fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=120)
    plt.hist(reprojection_error.ravel(), bins=np.linspace(0, 25, 50), density=True)
    plt.title("Reprojection Error Distribution Across All Views")
    plt.xlabel("Error (px)")
    plt.ylabel("PDF")
    plt.savefig(session / "reprojection_errors.png", format="png", dpi="figure")
    plt.close()
    return


def make_reprojection_imgs(session, detections, reprojections):
    """Make visualization of calibrated board corners."""
    pass


def generate_images(session, metadata):
    """Generate visualizations for inspection of calibration quality."""
    common_corners, _, reproj_corners = metadata
    reprojection_errors = np.linalg.norm(common_corners - reproj_corners, axis=-1)
    make_histogram(session, reprojection_errors)
    # make_reprojection_imgs(sess

    return


def collect_metadata(rows, cgroup):
    """Load metadata for calibration."""
    frames_per_view = [[x[i]["framenum"][1] for i in range(len(x))] for x in rows]
    common_frames = list(set.intersection(*[set(x) for x in frames_per_view]))
    idxs_per_view = [
        [x.index(frame) for frame in common_frames] for x in frames_per_view
    ]

    corners = [
        np.stack([x[i]["filled"].squeeze() for i in range(len(x))], axis=0)
        for x in rows
    ]
    common_corners = np.stack(
        [
            view_corners[view_idxs]
            for view_corners, view_idxs in zip(corners, idxs_per_view)
        ],
        axis=0,
    )
    corners_3d = cgroup.triangulate_optim(common_corners)
    corners_reproj = cgroup.project(corners_3d.reshape((-1, 2))).reshape(
        (len(corners), len(common_frames), -1, 2)
    )

    metadata = (common_corners, corners_3d, corners_reproj)
    generate_images(metadata)

    return metadata


def calibrate(session, board):
    """Calibrate cameras for a given session."""
    board_vids = [[x.as_posix()] for x in session.rglob("*.MOV")]

    cam_names = [x.name for x in session.iterdir() if x.is_dir()]
    cgroup = CameraGroup.from_names(cam_names)
    error, rows = cgroup.calibrate_videos(board_vids, board)
    cgroup.dump((session / "calibration.toml").as_posix())

    metadata = collect_metadata(rows, cgroup)

    return cgroup.get_dicts(), error, metadata
