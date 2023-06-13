"""Tests for triangulation."""

from sleap_anipose.triangulation import *
from pathlib import Path
from aniposelib.cameras import CameraGroup
import numpy as np
import h5py
import pytest


@pytest.mark.parametrize("frames,excluded_views,ransac", [((25, 75), ("side",), True)])
def test_triangulate(minimal_session, tmp_path, frames, excluded_views, ransac):
    calibration = Path(minimal_session) / "calibration.toml"
    assert calibration.exists()

    tmp_p3d = tmp_path / "p3d"
    tmp_p3d.mkdir()
    fname = tmp_p3d / "points3d.h5"

    p3d = triangulate(
        minimal_session,
        calibration.as_posix(),
        frames,
        excluded_views,
        ransac,
        fname.as_posix(),
    )

    # Testing shape of the output matrices.
    _, n_frames, n_tracks, n_nodes, _ = load_tracks(
        minimal_session, frames, excluded_views
    )[0].shape
    assert n_frames == frames[1] - frames[0]
    assert p3d.shape == (n_frames, n_tracks, n_nodes, 3)

    # Testing saving functionality.
    assert fname.exists()
    with h5py.File(fname, "r") as f:
        loaded_p3d = f["tracks"][:]
        loaded_frames = f["frames"][:]
    assert np.all(loaded_p3d == p3d)
    assert np.all(loaded_frames == frames)


@pytest.mark.parametrize("frames,excluded_views", [((25, 75), ("side",))])
def test_reproject(minimal_session, frames, excluded_views):
    assert (Path(minimal_session) / "points3d.h5").exists()
    assert (Path(minimal_session) / "calibration.toml").exists()
    with h5py.File(Path(minimal_session) / "points3d.h5", "r") as f:
        p3d = f["tracks"][:]
    cgroup = CameraGroup.load(Path(minimal_session) / "calibration.toml")

    cams = [x for x in cgroup.get_names() if x not in excluded_views]
    n_cams = len(cams)
    n_frames = frames[1] - frames[0]

    p2d = reproject(p3d, cgroup, frames, excluded_views)

    assert p2d.shape[0] == n_cams
    assert p2d.shape[1] == n_frames
    assert p2d.shape[-1] == 2


@pytest.mark.parametrize("frames,excluded_views", [((25, 75), ("side",))])
def test_load_tracks(minimal_session, frames, excluded_views):
    p2d, views = load_tracks(
        minimal_session, frames=frames, excluded_views=excluded_views
    )
    cams = CameraGroup.load((Path(minimal_session) / "calibration.toml")).get_names()
    cams = [Path(minimal_session) / x for x in cams if x not in excluded_views]
    assert cams == views
    assert p2d.shape[0] == len(cams)
    assert p2d.shape[1] == frames[1] - frames[0]
    assert p2d.shape[-1] == 2


@pytest.mark.parametrize("frames", [(25, 75)])
def test_load_view(minimal_session, frames):
    cams = [x.as_posix() for x in Path(minimal_session).iterdir() if x.is_dir()]
    shapes = []
    for cam in cams:
        p2d = load_view(cam, frames)
        shapes.append(p2d.shape)
    n_frames, n_tracks, n_nodes, _ = shapes[0]
    assert n_frames == frames[1] - frames[0]
    assert np.all([x == (n_frames, n_tracks, n_nodes, 2) for x in shapes])
