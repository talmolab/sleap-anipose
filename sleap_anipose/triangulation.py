"""This module contains utilities related to triangulation."""

import numpy as np
import h5py
import sleap
from pathlib import Path
from aniposelib.cameras import CameraGroup
from typing import Union
import click


def load_view(view: str) -> np.ndarray:
    """Load track for a view folder.

    Args:
        view: The path to the view folder (relative to the working directory).

    Returns:
        A (n_frames, n_tracks, n_nodes, 2) shape ndarray of the 2D points.
    """
    slp_file = list(Path(view).glob("*proofread.slp"))[0].as_posix()
    track = sleap.load_file(slp_file, detect_videos=True).numpy()
    return track


def load_tracks(session: str) -> np.ndarray:
    """Load all view tracks for a session folder.

    Args:
        session: The path pointing to the session directory (relative to the
            working directory).

    Returns:
        A (n_views, n_frames, n_tracks, n_nodes, 2) shape ndarray of the tracks.
    """
    views = [x for x in Path(session).iterdir() if x.is_dir()]
    tracks = np.stack([load_view(view) for view in views], axis=0)
    return tracks


def triangulate(
    p2d: Union[np.ndarray, str],
    calib: Union[CameraGroup, str],
    save: bool = False,
    session: str = ".",
    disp_progress: bool = False,
) -> np.ndarray:
    """Triangulate 3D points for a given session.

    Args:
        p2d: The path pointing to the session directory (relative to the
            working directory) or a (n_cams, n_frames, n_tracks, n_nodes, 2)
            array of the 2D points from different views.
        calib: The path pointing to the calibration file or the object
            containing the camera data. Note that the order of the cameras in
            the CameraGroup object must be the same as the order of the arrays
            along the camera axis.
        save: A flag determining whether or not to save the results of
            triangulation.
        session: The path to the session directory to save the points to. If
            not specified, will save to the current working directory.
        disp_progress: A flag determining whether or not to show triangulation
            progress, false by default.

    Returns:
        A matrix of shape (n_frames, n_tracks, n_nodes, 3) containing the
        triangulated 3D points. If save is True, the matrix will be saved to
        the session folder as 'points3d.h5'.
    """
    if type(p2d) == str:
        points_2d = load_tracks(p2d)
    else:
        points_2d = p2d.copy()

    if type(calib) == str:
        cgroup = CameraGroup.load(calib)
    else:
        cgroup = calib

    n_tracks = points_2d.shape[2]

    init_points = np.stack(
        [
            cgroup.triangulate_optim(
                points_2d[:, :, track], init_progress=disp_progress
            )
            for track in range(n_tracks)
        ],
        axis=1,
    )  # (n_frames, n_tracks, n_nodes, 3)

    points_3d = np.stack(
        [
            cgroup.optim_points(points_2d[:, :, track], init_points[:, track])
            for track in range(n_tracks)
        ],
        axis=1,
    )  # (n_frames, n_tracks, n_nodes, 3)

    if save:
        fname = Path(session) / "points3d.h5"
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "tracks",
                data=points_3d,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )
            f["tracks"].attrs[
                "Description"
            ] = "Shape: (n_frames, n_tracks, n_nodes, 3)."

    return points_3d


@click.command()
@click.option(
    "--p2d",
    help="Path pointing to the session directory containing the SLEAP track files.",
)
@click.option("--calib", help="Path pointing to the calibration.toml file.")
@click.option(
    "--save",
    default=False,
    help="Flag determining whether or not to save triangulation results.",
)
@click.option(
    "--session", default=".", help="Path to save the triangulation results to."
)
@click.option(
    "--disp_progress",
    default=False,
    help="Flag determining whether or not to display triangulation progress.",
)
def triangulate_cli(
    p2d: str,
    calib: str,
    save: bool = False,
    session: str = ".",
    disp_progress: bool = False,
) -> np.ndarray:
    return triangulate(p2d, calib, save, session, disp_progress)


def reproject(
    p3d: Union[np.ndarray, str],
    calib: Union[CameraGroup, str],
    save: bool = False,
    session: str = ".",
) -> np.ndarray:
    """Reproject triangulated points to each camera's view.

    Args:
        p3d: A (n_frames, n_tracks, n_nodes, 3) array of the triangulated 3D
            points or the path pointing to the h5 file containing the points.
        calib: An object containing all the camera calibration data or the path
            pointing to its saved file. The order of the cameras in this object
            determines the order of the reprojections along the cameras axis.
        save: A flag determining whether or not to save the reprojections.
        session: Path to the session containing the 3D points and calibration,
            assumed to be the working directory if not specified.

    Returns:
        A (n_cams, n_frames, n_tracks, n_nodes, 2) ndarray of the reprojections
        for each camera view. If the save flag is true, a (n_frames, n_tracks,
        n_nodes, 2) array of the corresponding reprojections will be saved to
        each view under the name 'reprojections.h5'.
    """
    if type(p3d) == str:
        with h5py.File(p3d, "r") as f:
            points = f["tracks"][:]
    else:
        points = p3d.copy()

    if type(calib) == str:
        cgroup = CameraGroup.load(calib)
    else:
        cgroup = calib

    n_frames, n_tracks, n_nodes, _ = points.shape
    cams = cgroup.get_names()

    reprojections = cgroup.project(p3d.reshape((-1, 3))).reshape(
        (len(cams), n_frames, n_tracks, n_nodes, 2)
    )

    if save:
        for i, cam in enumerate(cams):
            fname = Path(session) / cam / "reprojections.h5"
            with h5py.File(fname, "w") as f:
                f.create_dataset(
                    "tracks",
                    data=reprojections[i],
                    chunks=True,
                    compression="gzip",
                    compression_opts=1,
                )
                f["tracks"].attrs[
                    "Description"
                ] = f"Shape: (n_frames, n_tracks, n_nodes, 2). View: {cam}"

    return reprojections


@click.command()
@click.option("--p3d", help="Path pointing to the points_3d.h5 file.")
@click.option("--calib", help="Path pointing to the calibration.toml file.")
@click.option(
    "--save",
    default=False,
    help="Flag determining whether or not to save the reprojections.",
)
@click.option("--session", default=".", help="Path to save the reprojections to.")
def reproject_cli(
    p3d: str,
    calib: str,
    save: bool = False,
    session: str = ".",
) -> np.ndarray:
    return reproject(p3d, calib, save, session)
