"""Tests for triangulation."""

from sleap_anipose.triangulation import *
from pathlib import Path
from aniposelib.cameras import CameraGroup
import numpy as np
import h5py


def test_triangulate(minimal_session, tmp_path):
    calibration = Path(minimal_session) / "calibration.toml"
    assert calibration.exists()

    tmp_p3d = tmp_path / "p3d"
    tmp_p3d.mkdir()
    fname = tmp_p3d / "points3d.h5"

    p3d = triangulate(minimal_session, calibration, fname)

    # Testing shape of the output matrices.
    _, n_frames, n_tracks, n_nodes, _ = load_tracks(minimal_session).shape
    assert p3d.shape == (n_frames, n_tracks, n_nodes, 3)

    # Testing saving functionality.
    assert fname.exists()
    with h5py.File(fname, "r") as f:
        loaded_p3d = f["tracks"][:]
    assert np.all(loaded_p3d == p3d)


def test_reproject(minimal_session):
    assert (Path(minimal_session) / "points3d.h5").exists()
    with h5py.File(Path(minimal_session) / "points3d.h5", "r") as f:
        p3d = f["tracks"][:]
    assert (Path(minimal_session) / "calibration.toml").exists()
    cgroup = CameraGroup.load(Path(minimal_session) / "calibration.toml")

    cams = [x for x in Path(minimal_session).iterdir() if x.is_dir()]
    n_cams = len(cams)

    p2d = reproject(p3d, cgroup)

    assert p2d.shape[0] == n_cams
    assert p2d.shape[-1] == 2


def test_load_tracks(minimal_session):
    cams = [x for x in Path(minimal_session).iterdir() if x.is_dir()]
    p2d = load_tracks(minimal_session)
    assert p2d.shape[0] == len(cams)
    assert p2d.shape[-1] == 2


def test_load_view(minimal_session):
    cams = [x.as_posix() for x in Path(minimal_session).iterdir() if x.is_dir()]
    shapes = []
    for cam in cams:
        p2d = load_view(cam)
        shapes.append(p2d.shape)
    n_frames, n_tracks, n_nodes, _ = shapes[0]
    assert np.all([x == (n_frames, n_tracks, n_nodes, 2) for x in shapes])
