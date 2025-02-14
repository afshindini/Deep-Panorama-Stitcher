"""This a test for keypoint stitcher method"""

from pathlib import Path
import cv2
from panaroma_stitcher.keypoint_stitcher import KeypointStitcher


def test_detect_and_describe() -> None:
    """Test for detecting and describing keypoints in KeypointStitcher"""
    stitcher = KeypointStitcher(Path("./test_data/mountain"), feature_detector="sift")
    detector = stitcher.detect_and_describe()
    assert isinstance(detector, cv2.SIFT)
    stitcher = KeypointStitcher(Path("./test_data/mountain"), feature_detector="orb")
    detector = stitcher.detect_and_describe()
    assert isinstance(detector, cv2.ORB)
    stitcher = KeypointStitcher(Path("./test_data/mountain"), feature_detector="brisk")
    detector = stitcher.detect_and_describe()
    assert isinstance(detector, cv2.BRISK)


def test_matcher() -> None:
    """Test for matching keypoints in KeypointStitcher"""
    stitcher = KeypointStitcher(
        Path("./test_data/mountain"), feature_detector="sift", matcher_type="bf"
    )
    matcher = stitcher.matcher()
    assert isinstance(matcher, cv2.BFMatcher)
    stitcher = KeypointStitcher(
        Path("./test_data/mountain"), feature_detector="sift", matcher_type="flann"
    )
    matcher = stitcher.matcher()
    assert isinstance(matcher, cv2.FlannBasedMatcher)


def test_stitcher(tmp_path: Path) -> None:
    """Test for stitcher method in KeypointStitcher"""
    stitcher = KeypointStitcher(
        Path("./test_data/mountain"),
        feature_detector="sift",
        matcher_type="bf",
        number_feature=500,
    )
    stitcher.stitcher(str(tmp_path / "test_result.png"), True)
    assert Path(tmp_path / "test_result.png").exists()
