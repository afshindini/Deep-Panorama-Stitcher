"""This a detailed stitcher based on stitching library"""

from dataclasses import dataclass, field
from typing import Any, Mapping
import logging
import cv2
from stitching import Stitcher
from .utility import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class DetailedStitcher(ImageLoader):
    """Detailed stitcher class"""

    feature_number: int = field(default=500)
    detector_method: str = field(default="sift")
    matcher_type: str = field(default="homography")
    confidence_threshold: float = field(default=0.5)
    camera_estimator: str = field(default="homography")
    camera_adjustor: str = field(default="ray")

    def __post_init__(self) -> None:
        """Check if the matcher is defined or not and other post-processing requirements"""
        self.opencv_load_images()

    def _create_config(self) -> Mapping[str, Any]:
        return {
            "nfeatures": self.feature_number,
            "detector": self.detector_method,
            "matcher_type": self.matcher_type,
            "try_use_gpu": self.device == "cuda",
            "confidence_threshold": self.confidence_threshold,
            "estimator": self.camera_estimator,
            "adjuster": self.camera_adjustor,
        }

    def stitcher(self, result_path: str, framer: bool) -> Any:
        """Stitcher based on stitching library"""
        image_stitcher = Stitcher(**self._create_config())
        stitched_image = image_stitcher.stitch(self.images)
        logger.info("Stitching images was successful.")
        self.save_result(
            cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB), result_path, framer
        )
        return cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
