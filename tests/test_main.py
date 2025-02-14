"""Test for main console"""

from click.testing import CliRunner
from panaroma_stitcher.main import panaroma_stitcher_cli


def test_panaroma_stitcher_cli() -> None:
    """This is a test for main cli"""
    runner = CliRunner()
    result = runner.invoke(panaroma_stitcher_cli, ["--help"])
    assert result.exit_code == 0
    assert result


def test_kornia() -> None:
    """This is a test for kornia cli"""
    runner = CliRunner()
    result = runner.invoke(
        panaroma_stitcher_cli, ["--data_path", "test_data/mountain", "kornia"]
    )
    assert result.exit_code == 0
    assert result


def test_opencv_simple() -> None:
    """This is a test for opencv cli"""
    runner = CliRunner()
    result = runner.invoke(
        panaroma_stitcher_cli, ["--data_path", "test_data/mountain", "opencv-simple"]
    )
    assert result.exit_code == 0
    assert result


def test_keypoint_stitcher() -> None:
    """This is a test for keypoint stitcher cli"""
    runner = CliRunner()
    result = runner.invoke(
        panaroma_stitcher_cli,
        ["--data_path", "test_data/mountain", "keypoint-stitcher"],
    )
    assert result.exit_code == 0
    assert result


def test_detailed_stitcher() -> None:
    """This is a test for detailed stitcher cli"""
    runner = CliRunner()
    result = runner.invoke(
        panaroma_stitcher_cli,
        ["--data_path", "test_data/mountain", "detailed-stitcher"],
    )
    assert result.exit_code == 0
    assert result
