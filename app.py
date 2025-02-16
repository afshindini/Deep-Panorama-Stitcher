"""This is the code for only demo in hugging face. The mina code can be run as it is mentioned in readme."""

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import shutil
import tempfile
import gradio as gr

from src.panaroma_stitcher.kornia import KorniaStitcher
from src.panaroma_stitcher.opencv_simple import SimpleStitcher
from src.panaroma_stitcher.keypoint_stitcher import KeypointStitcher
from src.panaroma_stitcher.detailed_stitcher import DetailedStitcher
from src.panaroma_stitcher.sequential_stitcher import SequentialStitcher


@dataclass
class StitcherDemo:
    """This is a simple class for implementing demo in hugging face"""

    model_type: Any = field(init=False)

    def temp_dir(self, files: Any) -> str:
        """create temp folder for uploading the images in gradio"""
        temp_dir = Path(tempfile.gettempdir()) / "uploaded_images"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            file_name = Path(file).name
            dest_path = temp_dir / file_name
            shutil.move(file, str(dest_path))
        return str(temp_dir)

    def callback(self, files: Any) -> Any:
        """Callback function to be used within gradio"""
        print(files)
        print(self.model_type)
        input_dir = self.temp_dir(files)
        if self.model_type == "Simple Stitcher":
            stitcher1 = SimpleStitcher(
                image_dir=Path(input_dir), stitcher_type="panorama"
            )
            return stitcher1.stitcher()
        if self.model_type == "Detailed Stitcher":
            stitcher2 = DetailedStitcher(
                image_dir=Path(input_dir),
                feature_number=500,
                device="cpu",
                detector_method="sift",
                matcher_type="homography",
                confidence_threshold=0.05,
                camera_adjustor="ray",
                camera_estimator="homography",
            )
            return stitcher2.stitcher()
        if self.model_type == "Kornia Stitcher":
            stitcher3 = KorniaStitcher(image_dir=Path(input_dir))
            stitcher3.loftr_matcher(model="outdoor")
            return stitcher3.stitcher()
        if self.model_type == "Sequential Stitcher":
            stitcher4 = SequentialStitcher(
                image_dir=Path(input_dir),
                feature_detector="sift",
                matcher_type="bf",
                number_feature=500,
                final_size=(1000, 3000),
            )
            return stitcher4.stitcher()

        stitcher5 = KeypointStitcher(
            image_dir=Path(input_dir),
            feature_detector="sift",
            matcher_type="bf",
            number_feature=500,
        )
        return stitcher5.stitcher()

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
