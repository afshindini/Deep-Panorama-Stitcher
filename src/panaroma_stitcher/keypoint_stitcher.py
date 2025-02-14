"""This is a stitcher with sift descriptor and homography transformation"""

from dataclasses import dataclass, field
from typing import Any, Sequence, Optional
import logging
import cv2
import imutils
import numpy as np
import numpy.typing as npt

from .utility import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class KeypointStitcher(ImageLoader):
    """This is a pair-wise stitcher based on descriptor and ransac from opencv"""

    feature_detector: str = field(default="sift")
    number_feature: int = field(default=20)
    matcher_type: str = field(default="bf")

    def __post_init__(self) -> None:
        """Check if the matcher is defined or not and other post-processing requirements"""
        self.opencv_load_images()

    def detect_and_describe(self) -> Any:
        """Return the descriptors and key points of an image"""
        if self.feature_detector == "brisk":
            return cv2.BRISK.create()
        if self.feature_detector == "orb":
            return cv2.ORB.create(self.number_feature, edgeThreshold=7)
        return cv2.SIFT.create(
            self.number_feature, contrastThreshold=0.02, edgeThreshold=7, sigma=1.0
        )

    def matcher(self) -> Any:
        """Define matcher from opencv"""
        if self.matcher_type == "bf":
            if self.feature_detector == "sift":
                bruteforce = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            return bruteforce
        return cv2.FlannBasedMatcher({"algorithm": 0, "trees": 20}, {"checks": 150})

    def _stitcher_helper(
        self, image_right: npt.NDArray[np.float32], image_left: npt.NDArray[np.float32]
    ) -> npt.NDArray[Any]:
        """Define the helper for stitching images"""
        descriptor = self.detect_and_describe()
        img_right_key, img_right_desc = descriptor.detectAndCompute(image_right, None)
        img_left_key, img_left_desc = descriptor.detectAndCompute(image_left, None)
        matcher = self.matcher()
        matches = matcher.match(img_right_desc, img_left_desc)
        right_points = np.asarray(
            [img_right_key[match.queryIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        left_points = np.asarray(
            [img_left_key[match.trainIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        homography, _ = cv2.findHomography(right_points, left_points, cv2.RANSAC, 5.0)
        result = cv2.warpPerspective(
            image_right, homography, (image_left.shape[1] * 2, image_left.shape[0] * 2)
        )
        crds = self._boundary_cleaner_crds(result)
        result = self._boundary_cleaner(result, crds, image_left)
        return result

    @staticmethod
    def _boundary_cleaner_crds(result_image: npt.NDArray[Any]) -> Sequence[int]:
        """Select boundaries of stitched images without black area caused by warping images"""
        image_boarder = cv2.copyMakeBorder(
            result_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, (0, 0, 0)
        )  # type: ignore
        gray_boarder = cv2.cvtColor(image_boarder, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.threshold(gray_boarder, 0, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(
            thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresholded.shape, dtype="uint8")
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        min_rect = mask.copy()
        sub = mask.copy()
        while cv2.countNonZero(sub) > 0:
            min_rect = cv2.erode(min_rect, None)  # type: ignore
            sub = cv2.subtract(min_rect, thresholded)
        cnts = cv2.findContours(
            min_rect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)
        return cv2.boundingRect(c)

    @staticmethod
    def _boundary_cleaner(
        result_image: npt.NDArray[Any],
        crds: Sequence[int],
        left_image: Optional[npt.NDArray[Any]] = None,
    ) -> npt.NDArray[Any]:
        """Remove black boundaries of stitched images caused by warping images based on input crds"""
        if left_image is not None:
            temp = result_image.copy()
            result_image[0 : left_image.shape[0], 0 : left_image.shape[1]] = left_image
            result_image[crds[1] :, crds[0] :] = temp[crds[1] :, crds[0] :]
            return result_image
        return result_image[crds[1] : crds[1] + crds[3], crds[0] : crds[0] + crds[2]]

    def stitcher(self, result_path: str, framer: bool) -> None:
        """Stitch all the images together"""
        if len(self.images) == 0:
            logger.warning("No images to stitch.")
            return
        if len(self.images) == 1:
            logger.warning("The directory contains only one image.")
            return
        stitched_image = self._stitcher_helper(self.images[1], self.images[0])
        for idx in range(2, len(self.images)):
            temp = self._stitcher_helper(self.images[idx], stitched_image)
            stitched_image = temp
        self.save_result(
            cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB), result_path, framer
        )
