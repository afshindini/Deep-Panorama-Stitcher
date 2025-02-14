"""This stitches sequnces of images by finding homography transform between pairs of images"""

from dataclasses import dataclass, field
from typing import Any, Tuple

import logging
import cv2
import numpy as np
import numpy.typing as npt

from .utility import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class SequentialStitcher(ImageLoader):
    """This is pair wise stitcher between sequential images"""

    feature_detector: str = field(default="sift")
    number_feature: int = field(default=100)
    matcher_type: str = field(default="bf")
    final_size: Tuple[int, int] = field(default_factory=lambda: (1000, 1000))

    def __post_init__(self) -> None:
        """Check post-processing requirements"""
        self.opencv_load_images()

    def detect_and_describe(self) -> Any:
        """Return the descriptors and key points of an image"""
        if self.feature_detector == "brisk":
            return cv2.BRISK.create()
        if self.feature_detector == "orb":
            return cv2.ORB.create(self.number_feature, edgeThreshold=7)
        return cv2.SIFT.create(
            self.number_feature, contrastThreshold=0.01, edgeThreshold=7, sigma=0.8
        )

    def matcher(self) -> Any:
        """Matcher from opencv"""
        if self.matcher_type == "bf":
            if self.feature_detector == "sift":
                bruteforce = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            return bruteforce
        return cv2.FlannBasedMatcher({"algorithm": 0, "trees": 20}, {"checks": 150})

    @staticmethod
    def stitch_cleaner(
        img1: npt.NDArray[Any], img2: npt.NDArray[Any], thresh: int = 1
    ) -> npt.NDArray[Any]:
        """Paste the image at the correct position in the final stitched image without artifacts"""
        grey1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        _, mask1 = cv2.threshold(grey1, thresh, 255, cv2.THRESH_BINARY)
        mask1_inv = cv2.bitwise_not(mask1)
        im1 = cv2.bitwise_and(img1, img1, mask=mask1)
        im2 = cv2.bitwise_and(img2, img2, mask=mask1_inv)
        stitched_image = cv2.add(im1, im2)
        return stitched_image

    def _transform_finder(
        self, image_left: npt.NDArray[np.float32], image_right: npt.NDArray[np.float32]
    ) -> Any:
        """Define the helper for stitching images and return the homography between pairs of images"""
        descriptor = self.detect_and_describe()
        img_left_key, img_left_desc = descriptor.detectAndCompute(image_left, None)
        img_right_key, img_right_desc = descriptor.detectAndCompute(image_right, None)

        matcher = self.matcher()
        matches = matcher.match(img_right_desc, img_left_desc)
        right_points = np.asarray(
            [img_right_key[match.queryIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        left_points = np.asarray(
            [img_left_key[match.trainIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5.0)
        return homography

    def _apply_transform(
        self, img: npt.NDArray[Any], homography: npt.NDArray[np.float32]
    ) -> npt.NDArray[Any]:
        """Apply homography transform on an image"""
        return cv2.warpPerspective(
            img, homography, (self.final_size[1], self.final_size[0])
        )

    def stitcher(self, result_path: str, framer: bool) -> None:
        """Stitch all the images together two by two"""
        if len(self.images) == 0:
            logger.warning("No images to stitch.")
            return
        if len(self.images) == 1:
            logger.warning("The directory contains only one image.")
            return
        first_homography = np.array([[1.0, 0.0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]])
        result_prev = self._apply_transform(self.images[0], first_homography)
        for idx in range(1, len(self.images)):
            homography = self._transform_finder(self.images[idx - 1], self.images[idx])
            if idx == 1:
                temp_homography = np.dot(first_homography, homography)
                temp_result = self._apply_transform(self.images[idx], temp_homography)
            else:
                temp_homography = np.linalg.multi_dot([temp_homography, homography])
                temp_result = self._apply_transform(self.images[idx], temp_homography)
            result_prev = self.stitch_cleaner(temp_result, result_prev)
        self.save_result(
            cv2.cvtColor(result_prev, cv2.COLOR_BGR2RGB), result_path, framer
        )
