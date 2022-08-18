"""This module defines utilities related to calibration."""

import numpy as np
import matplotlib.pyplot as plt
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup
import h5py
from pathlib import Path
from typing import Tuple, List, Dict, Union
import imageio
from random import sample
import toml


def make_histogram(
    detections: np.ndarray,
    reprojections: np.ndarray,
    save: bool = False,
    session: str = ".",
):
    """Make a visualization histogram of the reprojection errors.

    Args:
        detections: A (n_cams, n_frames, n_corners, 2) array containing the
            detected corner positions for each camera view.
        reprojections: A (n_cams, n_frames, n_corners, 2) array containing the
            reprojected corner positions for each camera view.
        save: A flag determining whether or not to save the histogram
        session: The session directory to save the figure to, assumed to be
            the current working directory if not specified.
    """
    reprojection_error = np.linalg.norm((detections - reprojections), axis=-1)
    fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=120)

    plt.hist(reprojection_error.ravel(), bins=np.linspace(0, 25, 50), density=True)
    plt.title("Reprojection Error Distribution Across All Views")
    plt.xlabel("Error (px)")
    plt.ylabel("PDF")

    if save:
        plt.savefig(
            Path(session) / "reprojection_errors.png", format="png", dpi="figure"
        )


def make_reproj_imgs(
    detections: np.ndarray,
    reprojections: np.ndarray,
    frames: List[int],
    session: str = ".",
    n_samples=4,
    save: bool = False,
):
    """Make visualization of calibrated board corners.

    Args:
        detections: A (n_cams, n_frames, n_corners, 2) array of the detected
            corners.
        reprojections: A (n_cams, n_frames, n_corners, 2) array of the
            reprojected corners.
        frames: A length n_frames list of the frames visible from each view.
            These frames are used for triangulation and saved in metadata.
        n_samples: The number of images to make per view.
        session: Path containing the view subfolders with the calibration board
            images. Images are saved to the view subfolders in this folder as
            'session / view / reprojection-{frame}.jpg'. Assumed to be the
            current working directory if not specified.
        save: Flag determining whether or not to save images to the session.

    """
    cam_folders = [x for x in Path(session).iterdir() if x.is_dir()]
    sampled_frames = sample(frames, n_samples)

    for i, cam in enumerate(cam_folders):
        for frame in sampled_frames:
            img = imageio.imread(list(cam.glob(f"*/*{frame}.jpg"))[0])
            fig = plt.figure(figsize=(14, 12), dpi=120, facecolor="w")
            plt.scatter(
                detections[i, frames.index(frame), :, 0],
                detections[i, frames.index(frame), :, 1],
                s=300,
                color="r",
                marker="+",
                label="detections",
                lw=2.5,
            )
            plt.scatter(
                reprojections[i, frames.index(frame), :, 0],
                reprojections[i, frames.index(frame), :, 1],
                s=300,
                color="g",
                marker="x",
                label="reprojections",
                lw=2.5,
            )
            plt.imshow(img, cmap="gray")
            plt.xticks([])
            plt.yticks([])
            plt.legend()

            if save:
                fname = cam / f"reprojection-{frame}.png"
                plt.savefig(fname, format="png", dpi="figure")


def get_metadata(
    corner_data: List[List[Dict]],
    cgroup: CameraGroup,
    save: bool = False,
    session: str = ".",
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Generate metadata for calibration.

    Args:
        corner_data: A length n_cams list of nested lists of length n_frames.
            Each nested list is comprised of dictionaries organized as:
                'framenum': A tuple where the second entry is the frame number
                'corners': A (n_detected_frames, n_corners, 2) array of detected
                    corners
                'ids': A (n_corners, 1) array of the ids of each detected corner
                'filled': A (n_frames, n_corners, 2) array of the interpolated
                    corners
                'rvec': A rotation vector for the orientation of the image
                'tvec': A translation vector for the orientation of the image
        cgroup: The object containing all the relevant camera parameters.
        save: Flag determining whether to save the calibration metadata
            to the session. Files will be saved as
            'session / calibration_metadata.h5'.
        session: Path to save the metadata to. Assumed to be the current
            working directory if not specified.

    Returns:
        common_frames: A length n_frames list of the indices of calibration
            board images that had sufficient detections across all views. Saved
            under the 'frames' key in the metadata file.
        common_corners: A (n_cams, n_frames, n_corners, 2) array of the corner
            positions in each view for corners visible from all corners. Saved
            under the 'detected_corners' key in the metadata file.
        corner_3d: A (n_frames, n_corners, 3) array of the triangulated corners.
            Saved under the 'triangulated_corners' key in the metadata file.
        corners_reproj: A (n_cams, n_frames, n_corners, 2) array of the common
            corners reprojected into each view. Ordering of views same as
            the order found in the cgroup input. Saved under the
            'reprojected_corners' key in the metadata file.
    """
    frames_per_view = [
        [x[i]["framenum"][1] for i in range(len(x))] for x in corner_data
    ]

    common_frames = list(set.intersection(*[set(x) for x in frames_per_view]))

    idxs_per_view = [
        [x.index(frame) for frame in common_frames] for x in frames_per_view
    ]

    raw_detections = [
        np.stack([x[i]["filled"].squeeze() for i in range(len(x))], axis=0)
        for x in corner_data
    ]

    common_corners = np.stack(
        [
            view_corners[view_idxs]
            for view_corners, view_idxs in zip(raw_detections, idxs_per_view)
        ],
        axis=0,
    )

    corners_3d = cgroup.triangulate_optim(common_corners)
    corners_reproj = cgroup.project(corners_3d.reshape((-1, 2))).reshape(
        (len(raw_detections), len(common_frames), -1, 2)
    )

    if save:
        cam_names = cgroup.get_names()
        with h5py.File(Path(session) / "calibration.metadata.h5", "w") as f:
            f.create_dataset(
                "frames",
                data=common_frames,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )

            f.create_dataset(
                "detected_corners",
                data=common_corners,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )

            f.create_dataset(
                "triangulated_corners",
                data=corners_3d,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )

            f.create_dataset(
                "reprojected_corners",
                data=corners_reproj,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )

            f["frames"].attrs[
                "Description"
            ] = "The frames that were detected across all views."

            f["detected_corners"].attrs["Description"] = (
                "Shape: (n_cams, n_frames, n_corners, 2). " f"View order: {cam_names}."
            )

            f["triangulated_corners"].attrs[
                "Description"
            ] = "Shape: (n_frames, n_corners, 3)."

            f["reprojected_corners"].attrs["Description"] = (
                "Shape: (n_cams, n_corners, n_nodes, 2). " f"View order: {cam_names}."
            )

    metadata = (common_frames, common_corners, corners_3d, corners_reproj)
    return metadata


def make_calibration_videos(session: str):
    """Generate movies from calibration board images.

    Args:
        session: Path pointing to the session with the .
    """
    cams = [x for x in Path(session).iterdir() if x.is_dir()]

    for cam in cams:
        fname = (
            cam
            / "calibration_images"
            / f"{Path(session).name}-{cam.name}-calibration.MOV"
        )
        calibration_imgs = list(cam.glob("*/*.jpg"))
        writer = imageio.get_writer(fname, fps=30)

        for img in calibration_imgs:
            writer.append_data(imageio.imread(img))

        writer.close()


def read_board(board_file: str):
    """Read toml file detailing calibration board.

    Args:
        board_file: Path to the calibration board toml file.

    Returns:
        An aniposelib.CharucoBoard object containing the details of the board.
    """
    board_dict = toml.load(board_file)
    board = CharucoBoard(
        board_dict["board_width"],
        board_dict["board_height"],
        board_dict["square_length"],
        board_dict["marker_length"],
        board_dict["marker_bits"],
        board_dict["dict_size"],
    )
    return board


def write_board(
    fname: str,
    board_width: int,
    board_height: int,
    square_length: float,
    marker_length: float,
    marker_bits: int,
    dict_size: int,
):
    """Write a toml file detailing a calibration board.

    Args:
        fname: File name to save the board to, must end in .toml.
        board_width: Number of squares along the width of the board.
        board_height: Number of squares along the height of the board.
        square_length: Length of square edge in any measured units.
        marker_length: Length of marker edge in the same measured units as the
            square length.
        marker_bits: Number of bits encoded in the marker images.
        dict_size: Size of the dictionary used for marker encoding.
    """
    board_dict = {
        "board_width": board_width,
        "board_height": board_height,
        "square_length": square_length,
        "marker_length": marker_length,
        "marker_bits": marker_bits,
        "dict_size": dict_size,
    }
    with open(fname, "w") as f:
        toml.dump(board_dict, f)


def calibrate(
    session: str,
    board: Union[str, CharucoBoard, Dict],
    save_calib: bool = False,
    save_metadata: bool = False,
    histogram: bool = False,
    reproj_imgs: bool = False,
    save_hist: bool = False,
    save_reproj_imgs: bool = False,
) -> Tuple[CameraGroup, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate cameras for a given session.

    Args:
        session: Path pointing to the session to calibrate, must include the
            calibration board images in view subfolders.
        board: Either the path pointing to the board.toml file, the direct CharucoBoard
            object, or a dictionary with the following key / value pairs:
                'board_width': Number of squares along the width of the board.
                'board_height': Number of squares along the height of the board.
                'square_length': Length of square edge in any measured units.
                'marker_length': Length of marker edge in the same measured units as the
                    square length.
                'marker_bits': Number of bits encoded in the marker images.
                'dict_size': Size of the dictionary used for marker encoding.
        save_calib: Flag determining whether to save the calibration to the
            session.
        save_metadata: Flag determining whether to save the calibration metadata
            to the session.
        histogram: Flag determining whether or not to generate a histogram of
            the reprojection errors.
        reproj_imgs: Flag determining whether or not to generate overlaid
            images of the detected corners and reprojected corners.
        save_hist: Flag determining whether or not to save the histogram of
            reprojection errors.
        save_reproj_imgs: Flag determining whether or not to save the
            reprojection images.

    Returns:
        cgroup: A CameraGroup object containing all the camera parameters for
            each view. If save flag is set, saved as 'calibration.toml'.
        metadata: A tuple with the following entries:
            frames: A length n_frames list of the indices of calibration board
                images that had sufficient detections across all views.
            detections: A (n_cams, n_frames, n_corners, 2) array of the detected
                calibration board corners.
            triangulations: A (n_frames, n_corners, 3) array of the triangulated
                calibration board corners.
            reprojections: A (n_cams, n_frames, n_corners, 2) array of the
                reprojected calibration board corners.
    """
    board_vids = [x.as_posix() for x in Path(session).rglob("*.MOV")]

    if len(board_vids) == 0:
        make_calibration_videos(session)

    board_vids = [[x.as_posix()] for x in Path(session).rglob("*.MOV")]

    cam_names = [x.name for x in Path(session).iterdir() if x.is_dir()]
    cgroup = CameraGroup.from_names(cam_names)

    if type(board) == str:
        calib_board = read_board(board)
    elif type(board) == CharucoBoard:
        calib_board = board
    else:
        calib_board = CharucoBoard(
            board["board_width"],
            board["board_height"],
            board["square_length"],
            board["marker_length"],
            board["marker_bits"],
            board["dict_size"],
        )

    _, corners = cgroup.calibrate_videos(board_vids, calib_board)
    frames, detections, triangulations, reprojections = get_metadata(
        corners, cgroup, save_metadata, session
    )

    if histogram:
        make_histogram(detections, reprojections, save_hist, session)

    if reproj_imgs:
        make_reproj_imgs(detections, reprojections, frames, session, save_reproj_imgs)

    if save_calib:
        cgroup.dump((Path(session) / "calibration.toml").as_posix())

    metadata = (frames, detections, triangulations, reprojections)
    return (cgroup, metadata)
