"""Kornia stitcher"""

from dataclasses import dataclass, field
from typing import Any, Optional

import logging

import torch
import kornia as krn
import kornia.feature as krnfeat
from kornia.contrib import ImageStitcher
from .utility import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class KorniaStitcher(ImageLoader):
    """Kornia stitcher based on LoFTR"""

    matcher: Optional[Any] = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Check if the matcher is defined or not and other post-processing requirements"""
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.info("%s is not available", self.device)
            self.device = "cpu"
        self.kornia_load_images()

    def loftr_matcher(self, model: str = "outdoor") -> None:
        """define a feature matcher"""
        if self.device == "cuda":
            self.matcher = krnfeat.LoFTR(pretrained=model).cuda()
        else:
            self.matcher = krnfeat.LoFTR(pretrained=model)

    def local_matcher(
        self, number_of_features: int = 100, match_mode: str = "snn", thr: float = 0.8
    ) -> None:
        """Local feature matcher of Kornia. mathc_mode: snn, nn, mnn, smnn"""
        self.matcher = krnfeat.LocalFeatureMatcher(
            krnfeat.GFTTAffNetHardNet(number_of_features),
            krnfeat.DescriptorMatcher(match_mode, thr),
        )

    def keynote_matcher(
        self, number_of_features: int = 100, match_mode: str = "snn", thr: float = 0.8
    ) -> None:
        """KeyNet matcher"""
        self.matcher = krnfeat.LocalFeatureMatcher(
            krnfeat.KeyNetAffNetHardNet(number_of_features),
            krnfeat.DescriptorMatcher(match_mode, thr),
        )

    def stitcher(self, result_path: str) -> Any:
        """Stitch images with feature matcher"""
        if not self.matcher:
            raise ValueError("Kornia matcher is not defined. Use one of loftr_matcher")
        image_stitcher = ImageStitcher(self.matcher, estimator="ransac")
        with torch.no_grad():
            result = image_stitcher(*self.images)
        self.save_result(krn.tensor_to_image(result), result_path, False)  # type: ignore
        return krn.tensor_to_image(result)  # type: ignore[attr-defined]
