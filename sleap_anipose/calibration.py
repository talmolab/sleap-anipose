"""This module defines utilities related to calibration."""

import numpy as np
import matplotlib.pyplot as plt
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup
import h5py
import imageio
from pathlib import Path
from typing import Tuple, List, Dict, Union
from random import sample
import toml
import click
import cv2
from cv2 import aruco


def make_histogram(
    detections: np.ndarray,
    reprojections: np.ndarray,
    save_path: str = "",
):
    """Make a visualization histogram of the reprojection errors.

    Args:
        detections: A (n_cams, n_frames, n_corners, 2) array containing the
            detected corner positions for each camera view.
        reprojections: A (n_cams, n_frames, n_corners, 2) array containing the
            reprojected corner positions for each camera view.
        save_path: The path to save the figure to. Will only save the
            figure if a non-empty string is entered.
    """
    reprojection_error = np.linalg.norm((detections - reprojections), axis=-1)
    fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=120)

    plt.hist(reprojection_error.ravel(), bins=np.linspace(0, 25, 50), density=True)
    plt.title("Reprojection Error Distribution Across All Views")
    plt.xlabel("Error (px)")
    plt.ylabel("PDF")

    save = len(save_path) > 0

    if save:
        plt.savefig(save_path, format="png", dpi="figure")


def make_reproj_imgs(
    detections: np.ndarray,
    reprojections: np.ndarray,
    frames: List[int],
    session: str,
    excluded_views: Tuple[str] = (),
    n_samples=4,
    save_path: str = "",
):
    """Make visualization of calibrated board corners.

    Args:
        detections: A (n_cams, n_frames, n_corners, 2) array of the detected
            corners.
        reprojections: A (n_cams, n_frames, n_corners, 2) array of the
            reprojected corners.
        frames: A length n_frames list of the frames visible from each view.
            These frames are used for triangulation and saved in metadata.
        session: Path containing the view subfolders with the calibration board
            images.
        excluded_views: Names (not paths) of camera views to be excluded from
            reprojection. These views must have also been excluded from calibration.
            If not given, all views will be used.
        n_samples: The number of images to make per view.
        save_path: The session to save the images to. If not specified as a non-empty
            string, images will not be saved. Images are saved to the view subfolders in
            this folder as 'save_path / view / reprojection-{frame}.png'.
    """
    cam_folders = [
        x
        for x in Path(session).iterdir()
        if x.is_dir() and x.name not in excluded_views
    ]
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

            if len(save_path) > 0:
                fname = cam / f"reprojection-{frame}.png"
                plt.savefig(fname, format="png", dpi="figure")


def get_metadata(
    corner_data: List[List[Dict]], cgroup: CameraGroup, save_path: str = ""
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
        save_path: The file path to save the metadata to. Will only save if a non-empty
            string is entered.

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

        If the option to save is selected, all these values are saved in hdf5 file.
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

    if len(save_path) > 0:
        cam_names = cgroup.get_names()
        with h5py.File(save_path, "w") as f:
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


def make_calibration_videos(view: str):
    """Generate movies from calibration board images.

    Args:
        view: Path pointing to the view subfolder with the calibration board images.
    """
    session_name = Path(view).parent.name
    # TODO: add different movie extension functionality
    fname = view / "calibration_images" / f"{session_name}-{view.name}-calibration.MOV"
    calibration_imgs = list(Path(view).glob("*/*.jpg"))
    writer = imageio.get_writer(fname, fps=30)

    for img in calibration_imgs:
        writer.append(imageio.imread(img))

    writer.close()


@click.command()
@click.option(
    "--session", help="Path pointing to the session with the calibration board images."
)
def make_calibration_videos_cli(session: str):
    """Generate movies from calibration board images from the CLI."""
    make_calibration_videos(session)


def read_board(board_file: str):
    """Read toml file detailing calibration board.

    Args:
        board_file: Path to the calibration board toml file.

    Returns:
        An aniposelib.CharucoBoard object containing the details of the board.
    """
    board_dict = toml.load(board_file)
    board = CharucoBoard(
        board_dict["board_x"],
        board_dict["board_y"],
        board_dict["square_length"],
        board_dict["marker_length"],
        board_dict["marker_bits"],
        board_dict["dict_size"],
    )
    return board


@click.command()
@click.option("--board_file", help="Path to the calibration board toml file.")
def read_board_cli(board_file: str):
    """Read toml file detailing the calibration board from the CLI."""
    return read_board(board_file)


def write_board(
    board_name: str,
    board_x: int,
    board_y: int,
    square_length: float,
    marker_length: float,
    marker_bits: int,
    dict_size: int,
):
    """Write a toml file detailing a calibration board.

    Args:
        board_name: File name to save the board as.
        board_x: Number of squares along the width of the board.
        board_y: Number of squares along the height of the board.
        square_length: Length of square edge in any measured units.
        marker_length: Length of marker edge in the same measured units as the
            square length.
        marker_bits: Number of bits encoded in the marker images.
        dict_size: Size of the dictionary used for marker encoding.
    """
    board_dict = {
        "board_x": board_x,
        "board_y": board_y,
        "square_length": square_length,
        "marker_length": marker_length,
        "marker_bits": marker_bits,
        "dict_size": dict_size,
    }
    with open(board_name, "w") as f:
        toml.dump(board_dict, f)


@click.command()
@click.option("--board_name", help="File name to save the board as.")
@click.option("--board_x", help="Number of squares along the width of the board.")
@click.option("--board_y", help="Number of squares along the height of the board.")
@click.option("--square_length", help="Length of square edge in any units.")
@click.option(
    "--marker_length", help="Length of marker edge in same units as square length."
)
@click.option("--marker_bits", help="Number of bits encoded in the marker images.")
@click.option("--dict_size", help="Size of dictionary used for marking encoding.")
def write_board_cli(
    board_name: str,
    board_x: int,
    board_y: int,
    square_length: float,
    marker_length: float,
    marker_bits: int,
    dict_size: int,
):
    """Write a calibration board .toml file from the CLI."""
    write_board(
        board_name,
        board_x,
        board_y,
        square_length,
        marker_length,
        marker_bits,
        dict_size,
    )


def draw_board(
    board_name: str,
    board_x: int,
    board_y: int,
    square_length: float,
    marker_length: float,
    marker_bits: int,
    dict_size: int,
    img_width: int,
    img_height: int,
    save: str = "",
):
    """Draw and save a printable calibration board jpg file.

    Args:
        board_name: Path to save the file to.
        board_x: Number of squares along the width of the board.
        board_y: Number of squares along with height of the board.
        square_length: Length of square edges in meters.
        marker_length: Length of marker edges in meters.
        marker_bits: Number of bits in aruco markers.
        dict_size: Size of dictionary for encoding aruco markers.
        img_width: Width of the drawn board in pixels.
        img_height: Height of the drawn board in pixels.
        save: Path to the save the parameters of the board to. Will only save if a
            non-empty string is given.
    """
    ARUCO_DICTS = {
        (4, 50): aruco.DICT_4X4_50,
        (5, 50): aruco.DICT_5X5_50,
        (6, 50): aruco.DICT_6X6_50,
        (7, 50): aruco.DICT_7X7_50,
        (4, 100): aruco.DICT_4X4_100,
        (5, 100): aruco.DICT_5X5_100,
        (6, 100): aruco.DICT_6X6_100,
        (7, 100): aruco.DICT_7X7_100,
        (4, 250): aruco.DICT_4X4_250,
        (5, 250): aruco.DICT_5X5_250,
        (6, 250): aruco.DICT_6X6_250,
        (7, 250): aruco.DICT_7X7_250,
        (4, 1000): aruco.DICT_4X4_1000,
        (5, 1000): aruco.DICT_5X5_1000,
        (6, 1000): aruco.DICT_6X6_1000,
        (7, 1000): aruco.DICT_7X7_1000,
    }

    if (marker_bits, dict_size) not in ARUCO_DICTS.keys():
        raise Exception("Invalid marker bits or dictionary size.")
    else:
        aruco_dict = ARUCO_DICTS[(marker_bits, dict_size)]

    charuco_board = aruco.CharucoBoard_create(
        board_x, board_y, square_length, marker_length, aruco_dict
    )

    imboard = charuco_board.draw((img_width, img_height))
    cv2.imwrite(board_name, imboard)

    if len(save) > 0:
        write_board(
            save,
            board_x,
            board_y,
            square_length,
            marker_length,
            marker_bits,
            dict_size,
        )


@click.command()
@click.option("--board_name", help="Path to save the file to.")
@click.option("--board_x", help="Number of squares along the width of the board.")
@click.option("--board_y", help="Number of squares along the height of the board.")
@click.option("--square_length", help="Length of square edges in any units.")
@click.option(
    "--marker_length", help=("Length of marker edges in the units of square " "length.")
)
@click.option("--marker_bits", help="Number of bits in aruco markers.")
@click.option("--dict_size", help="Size of dictionary for encoding aruco markers.")
@click.option("--img_width", help="Width of the drawn image in pixels.")
@click.option("--img_height", help="Height of the drawn image in pixels.")
@click.option(
    "--save",
    show_default=True,
    default="",
    help=(
        "Path to the save the parameters of the board to. Only saves if a non-empty "
        "string is given."
    ),
)
def draw_board_cli(
    save_folder: str,
    board_x: int,
    board_y: int,
    square_length: float,
    marker_length: float,
    marker_bits: int,
    dict_size: int,
    img_width: int,
    img_height: int,
    save: str = "",
):
    """Draw and save a printable calibration board jpg file from the CLI."""
    draw_board(
        save_folder,
        board_x,
        board_y,
        square_length,
        marker_length,
        marker_bits,
        dict_size,
        img_width,
        img_height,
        save,
    )


def calibrate(
    session: str,
    board: Union[str, CharucoBoard, Dict],
    excluded_views: Tuple[str] = (),
    calib_fname: str = "",
    metadata_fname: str = "",
    histogram_path: str = "",
    reproj_path: str = "",
) -> Tuple[CameraGroup, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate cameras for a given session.

    Args:
        session: Path pointing to the session to calibrate, must include the
            calibration board images in view subfolders.
        board: Either the path pointing to the board.toml file, the direct CharucoBoard
            object, or a dictionary with the following key / value pairs:
                'board_x': Number of squares along the width of the board.
                'board_y': Number of squares along the height of the board.
                'square_length': Length of square edge in any measured units.
                'marker_length': Length of marker edge in the same measured units as the
                    square length.
                'marker_bits': Number of bits encoded in the marker images.
                'dict_size': Size of the dictionary used for marker encoding.
        excluded_views: Names (not paths) of camera views to be excluded from
            calibration. If not given, all views will be used.
        calib_fname: File path to save the calibration to (must end in .toml). Will not
            save unless a non-empty string is given.
        metadata_fname: File path to save the calibration metadata to (must end in .h5).
            Will not save unless a non-empty string is given.
        histogram_path: Path to save the histogram of reprojection errors to. Will not
            save unless a non-empty string is given.
        reproj_path: Path pointing to the session to save the board reprojection images
            to. Will not save unless a non-empty string is given.

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
    cams = [
        x
        for x in Path(session).iterdir()
        if x.is_dir() and x.name not in excluded_views
    ]
    cam_names = [x.name for x in cams]
    cgroup = CameraGroup.from_names(cam_names)

    calib_videos = []
    for cam in cams:
        calib_video = list(cam.glob("*/*.MOV"))
        if not calib_video:
            make_calibration_videos(cam.as_posix())
        calib_videos.append([calib_video[0].as_posix()])

    if type(board) == str:
        calib_board = read_board(board)
    elif type(board) == CharucoBoard:
        calib_board = board
    else:
        calib_board = CharucoBoard(
            board["board_x"],
            board["board_y"],
            board["square_length"],
            board["marker_length"],
            board["marker_bits"],
            board["dict_size"],
        )

    _, corners = cgroup.calibrate_videos(calib_videos, calib_board)
    frames, detections, triangulations, reprojections = get_metadata(
        corners, cgroup, metadata_fname
    )

    make_histogram(detections, reprojections, histogram_path)

    make_reproj_imgs(
        detections,
        reprojections,
        frames,
        session,
        excluded_views,
        n_samples=4,
        save_path=reproj_path,
    )

    if len(calib_fname) > 0:
        cgroup.dump(calib_fname)

    metadata = (frames, detections, triangulations, reprojections)
    return (cgroup, metadata)


@click.command()
@click.option("--session", help="Path pointing to the session to calibrate.")
@click.option("--board", help="Path pointing to the board.toml file.")
@click.option(
    "--calib_fname",
    default="",
    help=(
        "File path to save the calibration to. Will not save unless a non-empty "
        "string is given."
    ),
)
@click.option(
    "--metadata_fname",
    default="",
    help=(
        "File path to save the calibration metadata to. Will not save unless a "
        "non-empty string is given."
    ),
)
@click.option(
    "--histogram_path",
    default="",
    help=(
        "Path to save the histogram of reprojection errors to. Will not save unless a"
        " non-empty string is given."
    ),
)
@click.option(
    "--reproj_path",
    default="",
    help=(
        "Path pointing to the session to save the board reprojection images to. "
        "Will not save unless a non-empty string is given."
    ),
)
def calibrate_cli(
    session: str,
    board: str,
    calib_fname: str = "",
    metadata_fname: str = "",
    histogram_path: str = "",
    reproj_path: str = "",
) -> Tuple[CameraGroup, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate a session from the CLI."""
    return calibrate(
        session,
        board,
        calib_fname,
        metadata_fname,
        histogram_path,
        reproj_path,
    )
