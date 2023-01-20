"""This module contains utilities related to triangulation."""

from xml.dom import NotFoundErr
import numpy as np
import h5py
from pathlib import Path
from aniposelib.cameras import CameraGroup
from typing import Union, List, Tuple
import click


def load_view(view: str, frames: Tuple[int] = ()) -> np.ndarray:
    """Load track for a view folder.

    Args:
        view: The path to the view folder (relative to the working directory).
        frames: A tuple structured as (start_frame, end_frame) containing the frame
            range to load from the video. The range is (inclusive, exclusive) and will
            be considered as the entire video if not otherwise specified.

    Returns:
        A (n_frames, n_tracks, n_nodes, 2) shape ndarray of the 2D points.
    """
    h5_file = list(Path(view).glob("*analysis.h5"))[0].as_posix()
    with h5py.File(h5_file, "r") as f:
        track = f["tracks"][:].transpose((-1, 0, -2, 1))
    if frames:
        return track[frames[0] : frames[1]]
    else:
        return track


def load_tracks(
    session: str, frames: Tuple[int] = (), excluded_views: Tuple[str] = ()
) -> np.ndarray:
    """Load all view tracks for a session folder.

    Args:
        session: The path pointing to the session directory (relative to the
            working directory).
        frames: A tuple structured as (start_frame, end_frame) containing the frame
            range to load from each video. The range is (inclusive, exclusive) and will
            be considered as the entire video if not otherwise specified.
        excluded_views: Names (not paths) of camera views to be excluded. If non given,
            all views will be used.

    Returns:
        A (n_views, n_frames, n_tracks, n_nodes, 2) shape ndarray of the tracks.
    """
    views = [
        x
        for x in Path(session).iterdir()
        if x.is_dir() and x.name not in excluded_views
    ]
    tracks = np.stack([load_view(view, frames) for view in views], axis=0)
    return tracks


def triangulate(
    p2d: Union[np.ndarray, str],
    calib: Union[CameraGroup, str],
    frames: Tuple[int] = (),
    excluded_views: Tuple[str] = (),
    fname: str = "",
    disp_progress: bool = False,
    **kwargs,
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
        frames: A tuple structured as (start_frame, end_frame) containing the frame
            range to triangulate. The range is (inclusive, exclusive) and will be
            considered as the entire video if not otherwise specified.
        excluded_views: Names (not paths) of camera views to be excluded from
            triangulation. If not given, all views will be used.
        fname: The file path to save the triangulated points to (must end in .h5). Will
            not save unless a non-empty string is given.
        disp_progress: A flag determining whether or not to show triangulation
            progress, false by default.
        kwargs: Arguments related to filtering and limb constraints. Below are some of
            the more useful arguments, see the aniposelib for more details.
                constraints: A Kx2 array for rigid limb constraints, default empty. An
                    example would be [[0, 1], [2,3]], which denotes that the length
                    between joints 1 and 2 and the length between joints 2 and 3 are
                    constant.
                constraints_weak: A Kx2 array of more flexible constraints such as
                    shoulder length in humans or tarsus length in flies, default empty.
                scale_smooth: The weight of the temporal smoothing term in the loss
                    function, default 4.
                scale_length: The weight of the length constraints in the loss function,
                     default 2.
                scale_length_weak: The weight of the weak length constraints in the loss
                    function, default 0.5.
                reproj_error_threshold: A threshold for determining which points are not
                    suitable for triangulation, default 15.
                reproj_loss: The loss function for the reprojection loss, default
                    'soft_l1'. See scipy.optimize.least_squares for more options.
                n_deriv_smooth: The order of derivative to smooth for in the temporal
                    filtering, default 1.

    Returns:
        A matrix of shape (n_frames, n_tracks, n_nodes, 3) containing the triangulated
        3D points.
    """
    if type(p2d) == str:
        points_2d = load_tracks(p2d, frames, excluded_views)
    else:
        # TODO: Decouple view exclusion from input raw 2D coordinates
        if frames:
            points_2d = p2d.copy()[:, frames[0] : frames[1]]
        else:
            points_2d = p2d.copy()

    if type(calib) == str:
        full_cgroup = CameraGroup.load(calib)
    else:
        full_cgroup = calib

    if excluded_views:
        cgroup = full_cgroup.subset_cameras(
            [
                i
                for i, x in enumerate(full_cgroup.get_names())
                if x not in excluded_views
            ]
        )
    else:
        cgroup = full_cgroup

    n_tracks = points_2d.shape[2]

    if "constraints" in kwargs.keys():
        kwargs["constraints"] = (
            [] if kwargs["constraints"] is None else kwargs["constraints"]
        )
    if "constraints_weak" in kwargs.keys():
        kwargs["constraints_weak"] = (
            [] if kwargs["constraints_weak"] is None else kwargs["constraints_weak"]
        )

    points_3d = np.stack(
        [
            cgroup.triangulate_optim(
                points_2d[:, :, track], init_progress=disp_progress, **kwargs
            )
            for track in range(n_tracks)
        ],
        axis=1,
    )  # (n_frames, n_tracks, n_nodes, 3)

    if len(fname) > 0:
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "tracks",
                data=points_3d,
                chunks=True,
                compression="gzip",
                compression_opts=1,
            )

            if type(p2d) == str:
                cam_names = [
                    x.name
                    for x in Path(p2d).iterdir()
                    if x.is_dir() and x.name not in excluded_views
                ]
                tracks_descriptor = (
                    "Shape: (n_frames, n_tracks, n_nodes, 3). "
                    f"Camera views used: {cam_names}"
                )
            else:
                tracks_descriptor = "Shape: (n_frames, n_tracks, n_nodes, 3)."
            f["tracks"].attrs["Description"] = tracks_descriptor

            if frames:
                f.create_dataset(
                    "frames", data=frames, compression="gzip", compression_opts=1
                )
                f["frames"].attrs[
                    "Description"
                ] = "Range, inclusive to exclusive, of frames triangulated over."

    return points_3d


@click.command()
@click.option(
    "--p2d",
    type=str,
    required=True,
    help="Path pointing to the session directory containing the SLEAP track files.",
)
@click.option(
    "--calib", type=str, required=True, help="Path pointing to the calibration file."
)
@click.option(
    "--fname",
    type=str,
    required=True,
    help="The file path to save the triangulated points to (must end in .h5).",
)
@click.option(
    "--frames",
    nargs=2,
    type=int,
    default=(-1, -1),
    help="The range of frames (inclusive to exclusive) to triangulate over.",
)
@click.option(
    "--excluded_views",
    multiple=True,
    type=str,
    default=("ALL_VIEWS",),
    help=(
        "Names (not paths) of camera views to be excluded from triangulation. Specified"
        " via multiple calls, i.e. --excluded_views top --excluded_views side. If not "
        "specified, all views will be used."
    ),
)
@click.option(
    "--disp_progress",
    is_flag=True,
    default=False,
    help="Flag determining whether or not to display triangulation progress.",
)
@click.option(
    "--constraints",
    multiple=True,
    type=(int, int),
    default=((-1, -1),),
    help=(
        "Rigid limb constraints between different nodes. An example would be "
        "--constraints 0 1 --constraints 2 3, which would denote that the lengths "
        "between joints 0 and 1 and joints 2 and 3, respectively, are constant."
    ),
)
@click.option(
    "--constraints_weak",
    multiple=True,
    type=(int, int),
    default=((-1, -1),),
    help=(
        "Flexible constraints such as shoulder length in humans or tarsus length in "
        "flies. Uses same input pattern as the --constraints parameter."
    ),
)
@click.option(
    "--scale_smooth",
    type=float,
    default=4.0,
    help="The weight of the temporal smoothing term in the loss function, default 4.",
)
@click.option(
    "--scale_length",
    type=float,
    default=2.0,
    help="The weight of the length constraints in the loss function, default 2.",
)
@click.option(
    "--scale_length_weak",
    type=float,
    default=0.5,
    help="The weight of the weak length constraints in the loss function, default 2.",
)
@click.option(
    "--reproj_error_threshold",
    type=float,
    default=15.0,
    help="The threshold in pixels for discarding points for triangulation, default 15.",
)
@click.option(
    "--reproj_loss",
    type=str,
    default="soft_l1",
    help="Type of loss function for the reprojection error.",
)
@click.option(
    "--n_deriv_smooth",
    type=int,
    default=1,
    help="The order of derivative to smooth for in the temporal filtering, default 1.",
)
def triangulate_cli(
    p2d,
    calib,
    fname,
    frames,
    excluded_views,
    disp_progress,
    constraints,
    constraints_weak,
    scale_smooth,
    scale_length,
    scale_length_weak,
    reproj_error_threshold,
    reproj_loss,
    n_deriv_smooth,
):
    """Triangulate points from the CLI."""
    if frames == (-1, -1):
        frames = ()
    if excluded_views == ("ALL_VIEWS",):
        excluded_views = ()
    if constraints == ((-1, -1),):
        constraints = ()
    if constraints_weak == ((-1, -1),):
        constraints_weak = ()

    return triangulate(
        p2d,
        calib,
        frames,
        excluded_views,
        fname,
        disp_progress,
        constraints=constraints,
        constraints_weak=constraints_weak,
        scale_smooth=scale_smooth,
        scale_length=scale_length,
        scale_length_weak=scale_length_weak,
        reproj_error_threshold=reproj_error_threshold,
        reproj_loss=reproj_loss,
        n_deriv_smooth=n_deriv_smooth,
    )


def reproject(
    p3d: Union[np.ndarray, str],
    calib: Union[CameraGroup, str],
    frames: Tuple[int] = (),
    excluded_views: Tuple[str] = (),
    fname: str = "",
) -> np.ndarray:
    """Reproject triangulated points to each camera's view.

    Args:
        p3d: A (n_frames, n_tracks, n_nodes, 3) array of the triangulated 3D
            points or the path pointing to the h5 file containing the points.
        calib: An object containing all the camera calibration data or the path
            pointing to its saved file. The order of the cameras in this object
            determines the order of the reprojections along the cameras axis.
        frames: A tuple structured as (start_frame, end_frame) containing the frame
            range to reproject. The range is (inclusive, exclusive) and will be
            considered as the entire video if not otherwise specified.
        excluded_views: Names (not paths) of camera views to be excluded from
            reprojection. If not given, all views will be used.
        fname: The file path to save the reprojected points to (must end in .h5). Will
            not save unless a non-empty string is given.

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

    if frames:
        points = points[frames[0] : frames[1]]

    if type(calib) == str:
        full_cgroup = CameraGroup.load(calib)
    else:
        full_cgroup = calib

    if excluded_views:
        cgroup = full_cgroup.subset_cameras(
            [
                i
                for i, x in enumerate(full_cgroup.get_names())
                if x not in excluded_views
            ]
        )
    else:
        cgroup = full_cgroup

    n_frames, n_tracks, n_nodes, _ = points.shape
    cams = cgroup.get_names()

    reprojections = cgroup.project(points.reshape((-1, 3))).reshape(
        (len(cams), n_frames, n_tracks, n_nodes, 2)
    )

    if fname:
        with h5py.File(fname, "w") as f:
            for i, cam in enumerate(cams):
                f.create_dataset(
                    name=cam,
                    data=reprojections[i],
                    chunks=True,
                    compression="gzip",
                    compression_opts=1,
                )
                f[cam].attrs[
                    "Description"
                ] = f"Shape: (n_frames, n_tracks, n_nodes, 2)."

    return reprojections


@click.command()
@click.option(
    "--p3d",
    type=str,
    required=True,
    help="Path pointing to the .h5 file containing the 3D points.",
)
@click.option(
    "--calib", type=str, required=True, help="Path pointing to the calibration file."
)
@click.option(
    "--fname",
    type=str,
    required=True,
    help="The file path to save the reprojected points to (must end in .h5).",
)
@click.option(
    "--frames",
    nargs=2,
    type=int,
    default=(-1, -1),
    help="The range of frames (inclusive to exclusive) to reproject over.",
)
@click.option(
    "--excluded_views",
    multiple=True,
    type=str,
    default=("ALL_VIEWS",),
    help=(
        "Names (not paths) of camera views to be excluded from reprojection. Specified"
        " via multiple calls, i.e. --excluded_views top --excluded_views side. If not "
        "specified, all views will be used."
    ),
)
def reproject_cli(p3d, calib, fname, frames, excluded_views):
    """Triangulate points from the CLI."""
    if frames == (-1, -1):
        frames = ()
    if excluded_views == ("ALL_VIEWS",):
        excluded_views = ()
    return reproject(p3d, calib, frames, excluded_views, fname)
