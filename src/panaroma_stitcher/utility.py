"""Utility functions/classes"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any, Optional, Tuple

import logging
import torch
import cv2
import largestinteriorrectangle as lir
import kornia as krn
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class ImageLoader:
    """Load/Save images from/to directories"""

    image_dir: Path
    resize_shape: Optional[Tuple[int, int]] = field(default=None)
    device: str = field(default="cpu")
    images: List[Any] = field(init=False)

    def __post_init__(self) -> None:
        """Check the cuda availability and other post-processing requirements"""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.info("%s is not available.", self.device)
            self.device = "cpu"

    def _list_images(self) -> List[Path]:
        """List images in directory"""
        return sorted(
            filter(
                lambda path: path.suffix in [".jpg", ".png", ".tif"],
                self.image_dir.glob("*"),
            )
        )

    def opencv_load_images(self) -> None:
        """Load images for opencv stitcher from a directory"""
        files = self._list_images()
        if not self.resize_shape:
            self.images = [cv2.imread(str(filename)) for filename in files]
        else:
            self.images = [
                cv2.resize(cv2.imread(str(filename)), self.resize_shape)
                for filename in files
            ]
        logger.info(
            "Number of loaded images from %s is: %s",
            str(self.image_dir),
            len(self.images),
        )

    def kornia_load_images(self) -> None:
        """Load images for kornia stitcher from a directory"""
        files = list(self._list_images())
        if not self.resize_shape:
            self.images = [
                krn.io.load_image(
                    str(filename),
                    desired_type=krn.io.ImageLoadType.RGB32,
                    device=self.device,
                )[None, ...]
                for filename in files
            ]
        else:
            self.images = [
                krn.geometry.resize(
                    krn.io.load_image(
                        str(filename),
                        desired_type=krn.io.ImageLoadType.RGB32,
                        device=self.device,
                    )[None, ...],
                    self.resize_shape,
                )
                for filename in files
            ]
        logger.info(
            "Number of loaded images from %s is: %s",
            str(self.image_dir),
            len(self.images),
        )

    def remove_black_areas(self, img: Any) -> Any:
        """Remove black areas from stitched images"""
        image_boarder = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, (0, 0, 0))  # type: ignore
        gray_boarder = cv2.cvtColor(image_boarder, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.threshold(gray_boarder, 0, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(
            thresholded.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        contour = np.array([contours[0][:, 0, :]])
        inner_bb = lir.lir(contour)
        return img[
            inner_bb[1] : inner_bb[1] + inner_bb[3],
            inner_bb[0] : inner_bb[0] + inner_bb[2],
        ]

    def save_result(self, img: Any, save_path: str, framer: bool = True) -> None:
        """Save the final stitching result"""
        if framer:
            plt.imsave(save_path, self.remove_black_areas(img))
        else:
            plt.imsave(save_path, img)
