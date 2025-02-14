"""This is a test for kornia stitcher method"""

from pathlib import Path
import kornia.feature as krnfeat
from panaroma_stitcher.kornia import KorniaStitcher


def test_loftr_matcher() -> None:
    """Test for LOFTR matcher method in KorniaStitcher"""
    stitcher = KorniaStitcher(Path("./test_data/mountain"))
    stitcher.loftr_matcher("outdoor")
    assert isinstance(stitcher.matcher, krnfeat.LoFTR)
    stitcher.loftr_matcher("indoor")
    assert isinstance(stitcher.matcher, krnfeat.LoFTR)


def test_local_matcher() -> None:
    """Test for local matcher method in KorniaStitcher"""
    stitcher = KorniaStitcher(Path("./test_data/mountain"))
    stitcher.local_matcher()
    assert isinstance(stitcher.matcher, krnfeat.LocalFeatureMatcher)


def test_keynote_matcher() -> None:
    """Test for keynote matcher method in KorniaStitcher"""
    stitcher = KorniaStitcher(Path("./test_data/mountain"))
    stitcher.keynote_matcher()
    assert isinstance(stitcher.matcher, krnfeat.LocalFeatureMatcher)


def test_stitcher(tmp_path: Path) -> None:
    """Test for stitcher method in KorniaStitcher"""
    stitcher = KorniaStitcher(Path("./test_data/mountain"))
    stitcher.loftr_matcher("outdoor")
    stitcher.stitcher(str(tmp_path / "test_result.png"))
    assert Path(tmp_path / "test_result.png").exists()
