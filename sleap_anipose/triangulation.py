"""This module contains utilities related to triangulation."""

import numpy as np
import h5py
import sleap
from pathlib import Path
from aniposelib.cameras import CameraGroup


def load_view(view: Path) -> np.ndarray:
    """Load track for a view folder.

    Args:
        view: A path to the view folder.

    Returns:
        A (# frames, # tracks, # nodes, 2) shape ndarray of the 2D points.
    """
    slp_file = list(view.glob("*.proofread.slp"))[0].as_posix()
    track = sleap.load_file(slp_file, detect_videos=False).numpy()
    return track


def load_tracks(session):
    """Load all view tracks for a session folder.

    Args:
        session: A Path object pointing to the session directory (relative to the
            working directory)

    Returns:
        A (# views, # frames, # tracks, # nodes, 2) shape ndarray containing the 2D
        tracks.
    """
    views = [x for x in session.iterdir() if x.is_dir()]
    tracks = np.stack([load_view(view) for view in views], axis=0)
    return tracks


def triangulate(session, disp_progress=False):
    """Triangulate 3D points for a given session.

    Args:
        session: Path object or string pointing to the session directory (relative to
            the working directory).
        disp_progress: A boolean flag determining whether or not to show triangulation
            progress, False by default.

    Returns:
        points3d: The triangulated 3D points matrix, of shape
        `(n_frames, n_tracks, n_nodes, 3)`.
        Also saves the matrix to the session directory as points3d.h5.

    """
    if type(session) == str:
        session_path = Path(session)
    else:
        session_path = session

    # Grabbing calibration data
    cams = CameraGroup.load(session_path / "calibration.toml")

    # Grabbing 2D tracks
    points2d = load_tracks(session_path)  # (views, frames, tracks, nodes, 2)
    n_tracks = points2d.shape[2]

    # Triangulation
    init_points3d = np.stack(
        [
            cams.triangulate_optim(points2d[:, :, track], init_progress=disp_progress)
            for track in range(n_tracks)
        ],
        axis=1,
    )  # (frames, tracks, nodes, 3)
    points3d = np.stack(
        [
            cams.optim_points(points2d[:, :, track], init_points3d[:, track])
            for track in range(n_tracks)
        ],
        axis=1,
    )  # (frames, tracks, nodes, 3)

    # Saving file
    fname = session / "points3d.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset(
            "tracks", data=points3d, chunks=True, compression="gzip", compression_opts=1
        )
        f["tracks"].attrs[
            "Description"
        ] = "Array shape is (# frames, # tracks, # nodes, 3)."
    return points3d


def reproject(session):
    cgroup = CameraGroup.load(session / "calibration.toml")
    cams = cgroup.get_names()

    with h5py.File(session / "points3d.h5", "r") as f:
        p3d = f["tracks"][:]
    n_frames, n_tracks, n_nodes, _ = p3d.shape

    reprojections = cgroup.project_points(p3d.reshape((-1, 3))).reshape(
        (len(cams), n_frames, n_tracks, n_nodes, 2)
    )

    for i, cam in enumerate(cams):
        fname = session / cam / "reprojections.h5"
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
            ] = f"Array shape of (# frames, # tracks, # nodes, 2). View: {cam.name}."

    return reprojections
