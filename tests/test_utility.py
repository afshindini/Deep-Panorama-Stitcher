"""Unit Test for utility"""

from pathlib import Path
import pytest
import numpy as np
from panaroma_stitcher.utility import ImageLoader


@pytest.mark.parametrize(
    "data_path, num_images",
    [
        ("test_data/boat", 6),
        ("test_data/castle", 2),
        ("test_data/map", 6),
        ("test_data/mountain", 3),
        ("test_data/newspaper", 4),
        ("test_data/river", 3),
    ],
)
def test_opencv_load_images(data_path: str, num_images: int) -> None:
    """Unit test for opencv_load_images method of ImageLoader"""
    image_handler = ImageLoader(Path(data_path))
    image_handler.opencv_load_images()
    assert len(image_handler.images) == num_images


@pytest.mark.parametrize(
    "data_path, num_images",
    [
        ("test_data/boat", 6),
        ("test_data/castle", 2),
        ("test_data/map", 6),
        ("test_data/mountain", 3),
        ("test_data/newspaper", 4),
        ("test_data/river", 3),
    ],
)
def test_kornia_load_images(data_path: str, num_images: int) -> None:
    """Unit test for kornia_load_images method of ImageLoader"""
    image_handler = ImageLoader(Path(data_path))
    image_handler.kornia_load_images()
    assert len(image_handler.images) == num_images


def test_save_result(tmp_path: Path) -> None:
    """Unit test for save_result method of ImageLoader"""
    image_handler = ImageLoader(Path("./test_data/castle"))
    image_handler.save_result(
        np.ones([100, 100]), str(tmp_path / "test_result.png"), False
    )
    assert Path(tmp_path / "test_result.png").exists()
