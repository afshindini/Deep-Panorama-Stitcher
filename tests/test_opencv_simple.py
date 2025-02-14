"""This is a test for opencv-simple stitching method"""

from pathlib import Path
import pytest
from panaroma_stitcher.opencv_simple import SimpleStitcher


@pytest.mark.parametrize(
    "status_num, status",
    [
        (1, "ERR_NEED_MORE_IMGS"),
        (2, "ERR_HOMOGRAPHY_EST_FAIL"),
        (3, "ERR_CAMERA_PARAMS_ADJUST_FAIL"),
    ],
)
def test_stitching_status(status_num: int, status: str) -> None:
    """Test for stitching status method of SimpleStitcher"""
    stitcher = SimpleStitcher(Path("./test_data/mountain"))
    assert stitcher.stitching_status(status_num) == status


def test_stitcher(tmp_path: Path) -> None:
    """Test for stitching method of SimpleStitcher"""
    stitcher = SimpleStitcher(Path("./test_data/mountain"))
    stitcher.stitcher(str(tmp_path / "test_result.png"), True)
    assert Path(tmp_path / "test_result.png").exists()
