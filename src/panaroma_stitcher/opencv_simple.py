"""This is a simple stitcher approach with opencv"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import logging
import cv2
import torch
from .utility import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class SimpleStitcher(ImageLoader):
    """Simple stitcher approach with opencv"""

    stitcher_type: str = field(default="panorama")

    def __post_init__(self) -> None:
        """Check if the matcher is defined or not and other post-processing requirements"""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.info("%s is not available", self.device)
            self.device = "cpu"
        self.opencv_load_images()

    @staticmethod
    def stitching_status(status_number: int) -> Any:
        """Status for stitching"""
        return Enum(
            "stitch_status",
            [
                "ERR_NEED_MORE_IMGS",
                "ERR_HOMOGRAPHY_EST_FAIL",
                "ERR_CAMERA_PARAMS_ADJUST_FAIL",
            ],
        )(
            status_number
        ).name  # type: ignore

    def stitcher(self, result_path: str, framer: bool) -> Optional[Any]:
        """Stitch images with feature matcher"""
        if self.stitcher_type == "panorama":
            image_stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        elif self.stitcher_type == "scan":
            image_stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        else:
            logger.warning(
                "The stitcher_type %s is not defined in opencv library.",
                self.stitcher_type,
            )
            return None
        stitch_status, stitched_image = image_stitcher.stitch(self.images)
        if stitch_status == 0:
            logger.info("Stitching images was successful.")
            self.save_result(
                cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB), result_path, framer
            )
            return cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)

        logger.warning(
            "Stitching images FAILED with status: %s",
            self.stitching_status(stitch_status),
        )
        return None
