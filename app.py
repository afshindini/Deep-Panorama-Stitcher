"""This is the code for only demo in hugging face. The mina code can be run as it is mentioned in readme."""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import gradio as gr

from panaroma_stitcher import __version__
from panaroma_stitcher.kornia import KorniaStitcher
from panaroma_stitcher.opencv_simple import SimpleStitcher
from panaroma_stitcher.keypoint_stitcher import KeypointStitcher
from panaroma_stitcher.detailed_stitcher import DetailedStitcher
from panaroma_stitcher.sequential_stitcher import SequentialStitcher


@dataclass
class StitcherDemo:
    """This is a simple class for implementing demo in hugging face"""

    model_type: Any = field(init=False)

    def callback(self, files: Any) -> Any:
        """Callback function to be used within gradio"""
        input_dir = str(Path(files[0]).parent)
        if self.model_type == "Simple Stitcher":
            stitcher = SimpleStitcher(
                image_dir=Path(input_dir), stitcher_type="panorama"
            )
            return stitcher.stitcher("tmp/", True)
        if self.model_type == "Detailed Stitcher":
            stitcher = DetailedStitcher(
                image_dir=Path(input_dir),
                feature_number=500,
                device="cpu",
                detector_method="sift",
                matcher_type="homography",
                confidence_threshold=0.05,
                camera_adjustor="ray",
                camera_estimator="homography",
            )
            return stitcher.stitcher("tmp/", True)
        if self.model_type == "Kornia Stitcher":
            stitcher = KorniaStitcher(image_dir=Path(input_dir))
            stitcher.loftr_matcher(model="outdoor")
            return stitcher.stitcher("tmp/")
        if self.model_type == "Sequential Stitcher":
            stitcher = SequentialStitcher(
                image_dir=Path(input_dir),
                feature_detector="sift",
                matcher_type="bf",
                number_feature=500,
                final_size=(1000, 3000),
            )
            return stitcher.stitcher("tmp/", True)

        stitcher = KeypointStitcher(
            image_dir=Path(input_dir),
            feature_detector="sift",
            matcher_type="bf",
            number_feature=500,
        )
        return stitcher.stitcher("tmp/", True)

    def demo(self) -> None:
        """This is a design for the demo page"""
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    self.model_type = gr.Radio(
                        [
                            "Simple Stitcher",
                            "Detailed Stitcher",
                            "Kornia Stitcher",
                            "Sequential Stitcher",
                            "Keypoint Stitcher",
                        ],
                        label="Select the stitcher type",
                    )
                    files = gr.Files(file_types=["image"], file_count="multiple")

                    submit_btn = gr.Button(value="Stitch images")
                with gr.Column():
                    result = gr.Image(type="pil")
            submit_btn.click(  # pylint: disable=E1101
                self.callback, inputs=[files], outputs=result, api_name=False
            )
        demo.launch()


stitching_demo = StitcherDemo()
stitching_demo.demo()
