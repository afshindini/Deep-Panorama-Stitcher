"""Run the main code for panorama stitcher"""

from typing import Tuple, Any
from pathlib import Path
import logging
import click

from panaroma_stitcher import __version__
from panaroma_stitcher.logging import config_logger
from panaroma_stitcher.kornia import KorniaStitcher
from panaroma_stitcher.opencv_simple import SimpleStitcher
from panaroma_stitcher.keypoint_stitcher import KeypointStitcher
from panaroma_stitcher.detailed_stitcher import DetailedStitcher
from panaroma_stitcher.sequential_stitcher import SequentialStitcher

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Shorthand for info/debug/warning/error loglevel (-v/-vv/-vvv/-vvvv).",
)
@click.option("-r", "--resize_shape", type=(int, int), help="Shape to resize images.")
@click.option(
    "-d",
    "--data_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to data directory.",
)
@click.option(
    "-s",
    "--result_path",
    type=click.Path(),
    default=Path("./results.png"),
    help="Path to save the result file.",
)
@click.option(
    "--cleaner",
    is_flag=True,
    default=True,
    help="It crops the final image to remove the black background",
)
@click.pass_context
def panaroma_stitcher_cli(
    ctx: Any,
    verbose: int,
    resize_shape: Tuple[int],
    data_path: Path,
    result_path: Path,
    cleaner: bool,
) -> None:
    """This rep can stitch multi panorama images"""
    if verbose == 1:
        log_level = 10
    elif verbose == 2:
        log_level = 20
    elif verbose == 3:
        log_level = 30
    else:
        log_level = 40
    config_logger(log_level)
    ctx.ensure_object(dict)
    ctx.obj["resize_shape"] = resize_shape
    if resize_shape:
        logger.info(
            "Images from %s are shaped to %s for stitching.", data_path, resize_shape
        )
    ctx.obj["data_path"] = data_path
    ctx.obj["result_path"] = result_path
    ctx.obj["cleaner"] = cleaner


@panaroma_stitcher_cli.command()
@click.option(
    "--method",
    type=click.Choice(["loftr", "local", "keynote"], case_sensitive=False),
    default="loftr",
    help="Select the stitching method.",
)
@click.option(
    "--loftr_model",
    type=click.Choice(["outdoor", "indoor"], case_sensitive=False),
    default="outdoor",
    help="Select the pre-trained model in loftr method.",
)
@click.option(
    "--features",
    type=int,
    default=100,
    help="Number of features to in local/keynote methods",
)
@click.option(
    "--thr", type=float, default=0.8, help="Threshold for local/keynote method"
)
@click.option(
    "--matcher",
    type=click.Choice(["snn", "nn", "mnn", "smnn"], case_sensitive=False),
    default="snn",
    help="matcher mode in local/keynote methods.",
)
@click.pass_context
def kornia(
    ctx: Any, method: str, loftr_model: str, features: int, thr: float, matcher: str
) -> None:
    """This is cli for kornia stitcher techniques"""
    stitcher = KorniaStitcher(
        image_dir=Path(ctx.obj["data_path"]), resize_shape=ctx.obj["resize_shape"]
    )
    if method == "loftr":
        stitcher.loftr_matcher(model=loftr_model)
    if method == "local":
        stitcher.local_matcher(number_of_features=features, match_mode=matcher, thr=thr)
    if method == "keynote":
        stitcher.keynote_matcher(
            number_of_features=features, match_mode=matcher, thr=thr
        )
    _ = stitcher.stitcher(ctx.obj["result_path"])


@panaroma_stitcher_cli.command()
@click.option(
    "--stitcher_type",
    default="panorama",
    type=click.Choice(["scan", "panorama"], case_sensitive=False),
    help="Stitcher type for opencv stitching method.",
)
@click.pass_context
def opencv_simple(ctx: Any, stitcher_type: str) -> None:
    """This is cli for opencv simple stitcher"""
    stitcher = SimpleStitcher(
        image_dir=Path(ctx.obj["data_path"]),
        resize_shape=ctx.obj["resize_shape"],
        stitcher_type=stitcher_type,
    )
    _ = stitcher.stitcher(ctx.obj["result_path"], ctx.obj["cleaner"])


@panaroma_stitcher_cli.command()
@click.option(
    "--matching_method",
    default="bf",
    type=click.Choice(["bf", "flann"], case_sensitive=False),
    help="Choose matching type to be bruteforce/knn.",
)
@click.option(
    "--detector_method",
    default="sift",
    type=click.Choice(["sift", "orb", "brisk"], case_sensitive=False),
    help="Choose keypoint detection method.",
)
@click.option(
    "--number_feature",
    default=500,
    type=int,
    help="Number of features in detector methods.",
)
@click.pass_context
def keypoint_stitcher(
    ctx: Any, matching_method: str, detector_method: str, number_feature: int
) -> None:
    """This is cli for keypoint matching stitcher techniques"""
    stitcher = KeypointStitcher(
        image_dir=Path(ctx.obj["data_path"]),
        resize_shape=ctx.obj["resize_shape"],
        feature_detector=detector_method,
        matcher_type=matching_method,
        number_feature=number_feature,
    )
    _ = stitcher.stitcher(ctx.obj["result_path"], ctx.obj["cleaner"])


@panaroma_stitcher_cli.command()
@click.option(
    "--detect_method",
    default="sift",
    type=click.Choice(["sift", "orb", "brisk", "akaze"], case_sensitive=False),
    help="Choose keypoint detection method.",
)
@click.option(
    "--match_type",
    default="homography",
    type=click.Choice(["affine", "homography"], case_sensitive=False),
    help="Choose matching type to be affine/homography.",
)
@click.option(
    "--num_feat", default=500, type=int, help="Number of features in detector methods."
)
@click.option("--device", type=str, default="cpu", help="Use cuda/cpu")
@click.option(
    "--conf_thr",
    type=float,
    default=0.05,
    help="Confidence threshold for key point detection.",
)
@click.option(
    "--cam_est",
    default="homography",
    type=click.Choice(["affine", "homography"], case_sensitive=False),
    help="Choose camera estimator.",
)
@click.option(
    "--cam_adj",
    default="ray",
    type=click.Choice(["ray", "reproj", "affine", "no"], case_sensitive=False),
    help="Choose camera adjustor.",
)
@click.pass_context
def detailed_stitcher(  # pylint: disable=R0913, R0917
    ctx: Any,
    detect_method: str,
    match_type: str,
    num_feat: int,
    device: str,
    conf_thr: float,
    cam_est: str,
    cam_adj: str,
) -> None:
    """This is cli for detailed stitcher techniques from stitching library"""
    stitcher = DetailedStitcher(
        image_dir=Path(ctx.obj["data_path"]),
        resize_shape=ctx.obj["resize_shape"],
        feature_number=num_feat,
        device=device,
        detector_method=detect_method,
        matcher_type=match_type,
        confidence_threshold=conf_thr,
        camera_adjustor=cam_adj,
        camera_estimator=cam_est,
    )
    _ = stitcher.stitcher(ctx.obj["result_path"], ctx.obj["cleaner"])


@panaroma_stitcher_cli.command()
@click.option(
    "--matching_method",
    default="bf",
    type=click.Choice(["bf", "flann"], case_sensitive=False),
    help="Choose matching type to be bruteforce/knn.",
)
@click.option(
    "--detector_method",
    default="sift",
    type=click.Choice(["sift", "orb", "brisk"], case_sensitive=False),
    help="Choose keypoint detection method.",
)
@click.option(
    "--number_feature",
    default=500,
    type=int,
    help="Number of features in detector methods.",
)
@click.option("--final_shape", type=(int, int), help="Final result image shape.")
@click.pass_context
def sequential_stitcher(
    ctx: Any,
    matching_method: str,
    detector_method: str,
    number_feature: int,
    final_shape: Tuple[int, int],
) -> None:
    """This is cli for sequential stitcher techniques"""
    stitcher = SequentialStitcher(
        image_dir=Path(ctx.obj["data_path"]),
        resize_shape=ctx.obj["resize_shape"],
        feature_detector=detector_method,
        matcher_type=matching_method,
        number_feature=number_feature,
        final_size=final_shape,
    )
    _ = stitcher.stitcher(ctx.obj["result_path"], ctx.obj["cleaner"])
