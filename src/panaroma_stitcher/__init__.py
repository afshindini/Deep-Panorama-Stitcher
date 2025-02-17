""" This rep can stitch multi panaroma images """

__version__ = "0.2.0"

from .utility import ImageLoader
from .detailed_stitcher import DetailedStitcher
from .keypoint_stitcher import KeypointStitcher
from .kornia import KorniaStitcher
from .opencv_simple import SimpleStitcher

__all__ = [
    "ImageLoader",
    "DetailedStitcher",
    "KeypointStitcher",
    "KorniaStitcher",
    "SimpleStitcher",
]
