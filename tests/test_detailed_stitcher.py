"""This is a test for detailed stitcher"""

from pathlib import Path

from panaroma_stitcher.detailed_stitcher import DetailedStitcher


def test_stitcher(tmp_path: Path) -> None:
    """Test for detailed stitcher"""
    stitcher = DetailedStitcher(Path("./test_data/mountain"))
    stitcher.stitcher(str(tmp_path / "test_result.png"), True)
    assert Path(tmp_path / "test_result.png").exists()
