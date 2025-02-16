"""This is the code for only demo in hugging face. The mina code can be run as it is mentioned in readme."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
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

    parameters: Dict[str, Any] = field(init=False, default_factory=dict)
    param_values: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """initialize the parameters"""
        self.param_values["model"] = "Simple Stitcher"
        self.param_values["stitcher_type"] = "panorama"
        self.param_values["detect_method"] = "sift"
        self.param_values["match_type"] = "homography"
        self.param_values["num_feat"] = 500
        self.param_values["conf_thr"] = 0.05
        self.param_values["cam_est"] = "homography"
        self.param_values["cam_adj"] = "ray"
        self.param_values["matching_method"] = "bf"
        self.param_values["detector_method"] = "sift"
        self.param_values["number_feature"] = 500
        self.param_values["method"] = "loftr"
        self.param_values["loftr_model"] = "outdoor"
        self.param_values["features"] = 100
        self.param_values["matcher"] = "smnn"
        self.param_values["thr"] = 0.8

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

    def callback(self, files: Any) -> Optional[Any]:
        """Callback function to be used within gradio"""
        print(self.param_values)
        input_dir = self.temp_dir(files)
        if self.param_values["model"] == "Simple Stitcher":
            stitcher1 = SimpleStitcher(
                image_dir=Path(input_dir),
                stitcher_type=self.param_values["stitcher_type"],
            )
            return stitcher1.stitcher()
        if self.param_values["model"] == "Detailed Stitcher":
            stitcher2 = DetailedStitcher(
                image_dir=Path(input_dir),
                feature_number=self.param_values["num_feat"],
                device="cpu",
                detector_method=self.param_values["detect_method"],
                matcher_type=self.param_values["match_type"],
                confidence_threshold=self.param_values["conf_thr"],
                camera_adjustor=self.param_values["cam_adj"],
                camera_estimator=self.param_values["cam_est"],
            )
            return stitcher2.stitcher()
        if self.param_values["model"] == "Kornia Stitcher":
            stitcher3 = KorniaStitcher(image_dir=Path(input_dir))
            if self.param_values["method"] == "loftr":
                stitcher3.loftr_matcher(model=self.param_values["loftr_model"])
            if self.param_values["method"] == "local":
                stitcher3.local_matcher(
                    number_of_features=self.param_values["features"],
                    match_mode=self.param_values["matcher"],
                    thr=self.param_values["thr"],
                )
            if self.param_values["method"] == "keynote":
                stitcher3.keynote_matcher(
                    number_of_features=self.param_values["features"],
                    match_mode=self.param_values["matcher"],
                    thr=self.param_values["thr"],
                )
            return stitcher3.stitcher()
        if self.param_values["model"] == "Sequential Stitcher":
            stitcher4 = SequentialStitcher(
                image_dir=Path(input_dir),
                feature_detector=self.param_values["detector_method"],
                matcher_type=self.param_values["matching_method"],
                number_feature=self.param_values["number_feature"],
                final_size=(1000, 3000),
            )
            return stitcher4.stitcher()
        stitcher5 = KeypointStitcher(
            image_dir=Path(input_dir),
            feature_detector=self.param_values["detector_method"],
            matcher_type=self.param_values["matching_method"],
            number_feature=self.param_values["number_feature"],
        )
        return stitcher5.stitcher()

    def _design_simple_parameter(self) -> None:
        """Design the simple stitcher parameter section in gradio"""
        with gr.Accordion("Simple Stitcher Parameters", open=False):
            with gr.Row():
                self.parameters["stitcher_type"] = gr.Radio(
                    ["scan", "panorama"],
                    value=self.param_values["stitcher_type"],
                    label="Stitcher type",
                )

    def _design_detailed_parameter(self) -> None:
        """Design the detailed stitcher parameter section in gradio"""
        with gr.Accordion("Detailed Stitcher Parameters", open=False):
            with gr.Row():
                self.parameters["detect_method"] = gr.Radio(
                    ["sift", "orb", "brisk", "akaze"],
                    value=self.param_values["detect_method"],
                    label="Detection method",
                )
                self.parameters["match_type"] = gr.Radio(
                    ["affine", "homography"],
                    value=self.param_values["match_type"],
                    label="Matching method",
                )
                self.parameters["num_feat"] = gr.Number(
                    value=self.param_values["num_feat"],
                    precision=0,
                    label="No. of feature",
                )
                self.parameters["conf_thr"] = gr.Number(
                    value=self.param_values["conf_thr"], label="Confidence threshold"
                )
                self.parameters["cam_est"] = gr.Radio(
                    ["affine", "homography"],
                    value=self.param_values["cam_est"],
                    label="Camera estimator",
                )
                self.parameters["cam_adj"] = gr.Radio(
                    ["ray", "reproj", "affine", "no"],
                    value=self.param_values["cam_adj"],
                    label="Camera adjustor",
                )

    def _design_keypoint_parameter(self) -> None:
        """Design the keypoint/sequential stitcher parameter section in gradio"""
        with gr.Accordion("Keypoint/Sequential Stitcher Parameters", open=False):
            with gr.Row():
                self.parameters["matching_method"] = gr.Radio(
                    ["bf", "flann"],
                    value=self.param_values["matching_method"],
                    label="matching method",
                )
                self.parameters["detector_method"] = gr.Radio(
                    ["sift", "orb", "brisk"],
                    value=self.param_values["detector_method"],
                    label="detecting method",
                )
                self.parameters["number_feature"] = gr.Number(
                    value=self.param_values["number_feature"],
                    precision=0,
                    label="number of feature",
                )

    def _design_kornia_parameter(self) -> None:
        """Design the kornia stitcher parameter section in gradio"""
        with gr.Accordion("Kornia Stitcher Parameters", open=False):
            with gr.Row():
                self.parameters["method"] = gr.Radio(
                    ["loftr", "local", "keynote"],
                    value=self.param_values["method"],
                    label="Matching method",
                )
                self.parameters["loftr_model"] = gr.Radio(
                    ["outdoor", "indoor"],
                    value=self.param_values["loftr_model"],
                    label="loftr model type",
                )
                self.parameters["features"] = gr.Number(
                    value=self.param_values["features"],
                    precision=0,
                    label="No. of features in local/keynote methods",
                )
                self.parameters["matcher"] = gr.Radio(
                    ["snn", "nn", "mnn", "smnn"],
                    value=self.param_values["matcher"],
                    label="Matcher mode in local/keynote methods.",
                )
                self.parameters["thr"] = gr.Number(
                    value=self.param_values["thr"],
                    label="Threshold for local/keynote method",
                )

    def dummy_logger(self, val: Any, key: str) -> None:
        """Dummy logger to bes used in radio.change method"""
        # print(key, str(val))
        if isinstance(val, gr.components.radio.Radio):
            self.param_values[key] = str(val)
        if isinstance(val, gr.components.number.Number):
            if ((int(val) * 10) % 10) != 0:  # type: ignore
                self.param_values[key] = float(str(val))
            else:
                self.param_values[key] = int(str(val))

    def update_radio_button(self) -> None:
        """Need to update the radio options whenever we set default value for them"""
        for key, val in self.parameters.items():
            val.change(self.dummy_logger, inputs=[val, gr.State(key)], outputs=None)

    def demo(self) -> None:
        """This is a design for the demo page"""
        with gr.Blocks() as demo:
            # with gr.Row():
            with gr.Column():
                self.parameters["model"] = gr.Radio(
                    [
                        "Simple Stitcher",
                        "Kornia Stitcher",
                        "Sequential Stitcher",
                        "Keypoint Stitcher",
                        "Detailed Stitcher",
                    ],
                    value=self.param_values["model"],
                    label="Select the stitcher type",
                )
                self._design_simple_parameter()
                self._design_kornia_parameter()
                self._design_keypoint_parameter()
                self._design_detailed_parameter()
                self.update_radio_button()
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
