"""Tests for calibration."""

import numpy as np
from sleap_anipose.calibration import *
from pathlib import Path
from aniposelib.cameras import CameraGroup
import h5py
import toml


def test_calibrate(minimal_session, tmp_path):
    cam_names = [x.name for x in Path(minimal_session).iterdir() if x.is_dir()]
    board = read_board((Path(minimal_session) / "board.toml").as_posix())
    tmp_calib = tmp_path / "calibration"
    tmp_calib.mkdir()
    cgroup, _ = calibrate(minimal_session, board, True, tmp_calib.as_posix())

    # Testing the basics of the calibration.
    assert cgroup.get_names() == cam_names

    # Testing the shapes of the output matrices.
    intrinsics = [np.array(x["matrix"]) for x in cgroup.get_dicts()]
    rvecs = [x["rotation"] for x in cgroup.get_dicts()]
    tvecs = [x["translation"] for x in cgroup.get_dicts()]

    for p, r, t in zip(intrinsics, rvecs, tvecs):
        assert p.shape == (3, 3)
        assert len(r) == 3
        assert len(t) == 3

    # Testing the saving functionality.
    cgroup.dump(tmp_calib / "calibration.toml")
    loaded_cgroup = CameraGroup.load(tmp_calib / "calibration.toml")

    assert loaded_cgroup.get_names() == cgroup.get_names()

    loaded_intrinsics = [np.array(x["matrix"]) for x in loaded_cgroup.get_dicts()]
    loaded_rvecs = [x["rotation"] for x in loaded_cgroup.get_dicts()]
    loaded_tvecs = [x["translation"] for x in loaded_cgroup.get_dicts()]

    for i in range(len(loaded_intrinsics)):
        assert np.all(intrinsics[i] == loaded_intrinsics[i])
        assert np.all(rvecs[i] == loaded_rvecs[i])
        assert np.all(tvecs[i] == loaded_tvecs[i])


def test_get_metadata(minimal_session, tmp_path):
    cam_names = [x.name for x in Path(minimal_session).iterdir() if x.is_dir()]
    cgroup = CameraGroup.from_names(cam_names)

    board_vids = [[x.as_posix()] for x in list(Path(minimal_session).rglob("*.MOV"))]
    board = read_board((Path(minimal_session) / "board.toml").as_posix())

    tmp_calib = tmp_path / "calib_meta"
    tmp_calib.mkdir()

    _, corner_data = cgroup.calibrate_videos(board_vids, board, verbose=False)
    frames, detections, triangulations, reprojections = get_metadata(
        corner_data, cgroup, save=True, session=tmp_calib.as_posix()
    )

    # Testing saving functionality.
    assert (tmp_calib / "calibration.metadata.h5").exists()
    loading_path = tmp_calib / "calibration.metadata.h5"

    # Testing the shapes of output matrices.
    board_width, board_height = board.get_size()
    n_frames = len(frames)
    n_corners = (board_height - 1) * (board_width - 1)
    n_cams = len(cam_names)

    assert detections.shape == (n_cams, n_frames, n_corners, 2)
    assert triangulations.shape == (n_frames, n_corners, 3)
    assert reprojections.shape == (n_cams, n_frames, n_corners, 2)

    # Testing contents of the saved file.
    with h5py.File(loading_path, "r") as f:
        loaded_frames = f["frames"][:]
        loaded_detections = f["detected_corners"][:]
        loaded_triangulations = f["triangulated_corners"][:]
        loaded_reprojections = f["reprojected_corners"][:]

    assert np.all(loaded_frames == frames)
    assert np.all(
        loaded_detections[~np.isnan(loaded_detections)]
        == detections[~np.isnan(detections)]
    )
    assert np.all(
        loaded_triangulations[~np.isnan(loaded_triangulations)]
        == triangulations[~np.isnan(triangulations)]
    )
    assert np.all(
        loaded_reprojections[~np.isnan(loaded_reprojections)]
        == reprojections[~np.isnan(reprojections)]
    )


def test_read_board(minimal_session):
    board_path = Path(minimal_session) / "board.toml"
    assert board_path.exists()
    board = read_board(board_path.as_posix())
    file_dict = toml.load(board_path)

    assert board.get_size() == (file_dict["board_width"], file_dict["board_height"])
    assert board.get_square_length() == file_dict["square_length"]


def test_write_board(tmp_path):
    tmp_board = tmp_path / "boards"
    tmp_board.mkdir()
    board_path = (tmp_board / "board.toml").as_posix()
    board_width = 0
    board_height = 0
    square_length = 0.0
    marker_length = 0.0
    marker_bits = 0
    dict_size = 0
    write_board(
        board_path,
        board_width,
        board_height,
        square_length,
        marker_length,
        marker_bits,
        dict_size,
    )
    assert Path(board_path).exists()


# def test_make_histogram():
#     pass
#
# def test_make_reproj_imgs():
#     pass


# def test_make_calibration_videos(minimal_session, tmp_path):
#     video_folder = tmp_path / 'calib_videos'
#     video_folder.mkdir()

#     cam_folders = [x for x in minimal_session.iterdir() if x.is_dir()]

#     for cam in cam_folders:
