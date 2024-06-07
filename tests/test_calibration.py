"""Tests for calibration."""

import toml
from pathlib import Path

import h5py
import numpy as np
import pytest
from aniposelib.boards import CalibrationObject
from aniposelib.cameras import CameraGroup
from click.testing import CliRunner

from sleap_anipose.calibration import *


@pytest.mark.parametrize("excluded_views", [("side",)])
def test_calibrate(minimal_session, tmp_path, excluded_views):
    board = read_board((Path(minimal_session) / "board.toml").as_posix())
    tmp_calib = tmp_path / "calibration"
    tmp_calib.mkdir()
    save_path = (tmp_calib / "calibration.toml").as_posix()
    cgroup = calibrate(minimal_session, board, excluded_views, save_path)

    # Testing the basics of the calibration.
    cam_names = sorted(
        [
            x.name
            for x in Path(minimal_session).iterdir()
            if x.is_dir() and x.name not in excluded_views
        ]
    )
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
    loaded_cgroup = CameraGroup.load(save_path)

    assert loaded_cgroup.get_names() == cgroup.get_names()

    loaded_intrinsics = [np.array(x["matrix"]) for x in loaded_cgroup.get_dicts()]
    loaded_rvecs = [x["rotation"] for x in loaded_cgroup.get_dicts()]
    loaded_tvecs = [x["translation"] for x in loaded_cgroup.get_dicts()]

    for i in range(len(loaded_intrinsics)):
        assert np.all(intrinsics[i] == loaded_intrinsics[i])
        assert np.all(rvecs[i] == loaded_rvecs[i])
        assert np.all(tvecs[i] == loaded_tvecs[i])


@pytest.mark.parametrize("excluded_views", [("side",)])
def test_make_reproj_imgs(minimal_session, tmp_path, excluded_views):
    board = read_board((Path(minimal_session) / "board.toml").as_posix())
    tmp_calib = tmp_path / "calibration"
    tmp_calib.mkdir()

    # Testing the creation of reprojection images in each camera view folder
    cam_names = [
        x.name
        for x in Path(minimal_session).iterdir()
        if x.is_dir() and x.name not in excluded_views
    ]

    for cam in cam_names:
        if cam not in excluded_views:
            tmp_cam = tmp_calib / cam
            tmp_cam.mkdir()

    save_path = (Path(tmp_calib) / "calibration.toml").as_posix()
    cgroup = calibrate(minimal_session, board, excluded_views, save_path)

    calib_videos = []
    for cam in cam_names:
        cam_path = Path(minimal_session) / cam  # Ensure cam is treated as Path
        calib_video = list(cam_path.glob("*/*calibration.mp4"))
        if calib_video:  # Check if the list is not empty
            calib_videos.append([calib_video[0].as_posix()])

    _, corners = cgroup.calibrate_videos(calib_videos, board)

    metadata_fname = (Path(tmp_calib) / "calibration.metadata.h5").as_posix()

    frames, detections, triangulations, reprojections = get_metadata(
        corners, cgroup, metadata_fname
    )

    make_reproj_imgs(
        detections,
        reprojections,
        frames,
        minimal_session,
        excluded_views,
        n_samples=4,
        save_path=tmp_calib,
    )

    reproj_images = []
    for cam_name in cam_names:
        cam_path = tmp_calib / cam_name
        cam_reproj_images = list(cam_path.glob("reprojection-*.png"))
        reproj_images.extend(cam_reproj_images)
        print(f"Reprojection images found in {tmp_calib}: {cam_reproj_images}")

    assert len(reproj_images) > 0, "No reprojection images were created."
    assert (
        len(reproj_images) == 12
    ), "Expected number of reprojection images does not match."


@pytest.mark.parametrize("excluded_views", [("side",)])
def test_get_metadata(minimal_session, tmp_path, excluded_views):
    cams = [
        x
        for x in Path(minimal_session).iterdir()
        if x.is_dir() and x.name not in excluded_views
    ]
    cgroup = CameraGroup.from_names([x.name for x in cams])

    board_vids = [[list(cam.glob("*/*calibration.mp4"))[0].as_posix()] for cam in cams]
    board = read_board((Path(minimal_session) / "board.toml").as_posix())

    tmp_calib = tmp_path / "calib_meta"
    tmp_calib.mkdir()
    fname = tmp_calib / "calibration.metadata.h5"

    _, corner_data = cgroup.calibrate_videos(board_vids, board, verbose=False)
    frames, detections, triangulations, reprojections = get_metadata(
        corner_data, cgroup, fname.as_posix()
    )

    # Testing saving functionality.
    assert fname.exists()

    # Testing the shapes of output matrices.
    board_width, board_height = board.get_size()
    n_frames = len(frames)
    n_corners = (board_height - 1) * (board_width - 1)
    n_cams = len(cams)

    assert detections.shape == (n_cams, n_frames, n_corners, 2)
    assert triangulations.shape == (n_frames, n_corners, 3)
    assert reprojections.shape == (n_cams, n_frames, n_corners, 2)

    # Testing contents of the saved file.
    with h5py.File(fname, "r") as f:
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


def read_board_and_assert(board_path: str, is_charuco: bool):
    """Helper function for testing read_board."""
    board = read_board(board_path)
    file_dict = toml.load(board_path)

    assert board.squaresX == file_dict["board_x"]
    assert board.squaresY == file_dict["board_y"]
    assert board.square_length == file_dict["square_length"]

    if is_charuco:
        assert isinstance(board, CharucoBoard)
        assert board.marker_length == file_dict["marker_length"]
    else:
        assert isinstance(board, Checkerboard)


@pytest.mark.parametrize("board_toml", ["board.toml", "board_checker.toml"])
def test_read_board(minimal_session, board_toml):

    board_path: Path = Path(minimal_session) / board_toml
    read_board_and_assert(
        board_path=board_path.as_posix(), is_charuco=board_toml == "board.toml"
    )


@pytest.mark.parametrize("board_toml", ["board.toml", "board_checker.toml"])
def test_write_board(tmp_path, board_toml):
    tmp_board = tmp_path / "boards"
    tmp_board.mkdir()
    board_path = (tmp_board / board_toml).as_posix()
    board_x = 2
    board_y = 2
    square_length = 3.0

    if board_toml == "board.toml":
        marker_length = 2.0
        marker_bits = 5
        dict_size = 50
    else:
        marker_length = None
        marker_bits = None
        dict_size = None

    write_board(
        board_name=board_path,
        board_x=board_x,
        board_y=board_y,
        square_length=square_length,
        marker_length=marker_length,
        marker_bits=marker_bits,
        dict_size=dict_size,
    )
    assert Path(board_path).exists()

    read_board_and_assert(board_path=board_path, is_charuco=board_toml == "board.toml")


@pytest.mark.parametrize(
    "is_charuco",
    [True, False],
)
@pytest.mark.parametrize(
    "board_type",
    [str, CalibrationObject, dict],
)
def test_determine_board_type(tmp_path, is_charuco, board_type):

    # Determin which board dict to use as ground truth.
    board_dict = dict(
        board_x=2,
        board_y=2,
        square_length=3.0,
    )
    charuco_only_dict = dict(
        marker_length=2.0,
        marker_bits=5,
        dict_size=50,
    )
    if is_charuco:
        board_dict.update(charuco_only_dict)
        board_obj = CharucoBoard
    else:
        board_dict = board_dict
        board_obj = Checkerboard

    # Write the board to a file if it is a str.
    if board_type is str:
        board = (tmp_path / "board.toml").as_posix()
        write_board(board, **board_dict)

    # Use the board dict if it is a dict.
    elif board_type is dict:
        board = board_dict

    # Use the board object if it is a CalibrationObject.
    else:
        squaresX = board_dict.pop("board_x")
        squaresY = board_dict.pop("board_y")
        board = board_obj(squaresX=squaresX, squaresY=squaresY, **board_dict)

    calib_board = determine_board_type(board)
    assert isinstance(calib_board, board_obj)


@pytest.mark.parametrize("board_toml", ["board.toml", "board_checker.toml"])
def test_write_board_cli(tmp_path, board_toml):
    tmp_board = tmp_path / "boards"
    tmp_board.mkdir()
    board_path = (tmp_board / board_toml).as_posix()
    board_x = 2
    board_y = 2
    square_length = 3.0

    if board_toml == "board.toml":
        marker_length = 2.0
        marker_bits = 5
        dict_size = 50
    else:
        marker_length = None
        marker_bits = None
        dict_size = None

    runner = CliRunner()
    args = [
        "--board_name",
        board_path,
        "--board_x",
        board_x,
        "--board_y",
        board_y,
        "--square_length",
        square_length,
    ]

    if board_toml == "board.toml":
        args.extend(
            [
                "--marker_length",
                marker_length,
                "--marker_bits",
                marker_bits,
                "--dict_size",
                dict_size,
            ]
        )

    result = runner.invoke(
        write_board_cli,
        args,
    )
    assert result.exit_code == 0

    assert Path(board_path).exists()
    read_board_and_assert(board_path=board_path, is_charuco=board_toml == "board.toml")
