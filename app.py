# import numpy as np
# from PIL import Image
# from huggingface_hub import snapshot_download
# from leffa.transform import LeffaTransform
# from leffa.model import LeffaModel
# from leffa.inference import LeffaInference
# from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
# from leffa_utils.densepose_predictor import DensePosePredictor
# from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
# from preprocess.humanparsing.run_parsing import Parsing
# from preprocess.openpose.run_openpose import OpenPose
# import gradio as gr

import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
# from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
# Removing preprocess_garment_image since it's not found in utils.
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Download checkpoints
print("Downloading checkpoints...")
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")
print("Download completed.")

class LeffaPredictor(object):
    def __init__(self):
        print("Initializing LeffaPredictor components...")
        # print("Initializing mask predictor...")
        # self.mask_predictor = AutoMasker(
        #     densepose_path="./ckpts/densepose",
        #     schp_path="./ckpts/schp",
        # )

        print("Initializing DensePose predictor...")
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        print("Initializing human parsing...")
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        print("Initializing OpenPose...")
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        print("Loading VITON-HD model...")
        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)
        print("VITON-HD model loaded.")

        # The following models and resources are commented out to save memory:
        # print("Loading DressCode model...")
        # vt_model_dc = LeffaModel(
        #     pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
        #     pretrained_model="./ckpts/virtual_tryon_dc.pth",
        #     dtype="float16",
        # )
        # self.vt_inference_dc = LeffaInference(model=vt_model_dc)
        # print("DressCode model loaded.")

        # print("Loading Pose Transfer model...")
        # pt_model = LeffaModel(
        #     pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
        #     pretrained_model="./ckpts/pose_transfer.pth",
        #     dtype="float16",
        # )
        # self.pt_inference = LeffaInference(model=pt_model)
        # print("Pose Transfer model loaded.")
        print("LeffaPredictor initialization complete.")

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False
    ):
        print("Starting prediction...")
        try:
            print("Opening source image:", src_image_path)
            src_image = Image.open(src_image_path)
            print("Resizing source image...")
            src_image = resize_and_center(src_image, 768, 1024)
        except Exception as e:
            print("Error loading or resizing source image:", e)
            raise

        try:
            print("Processing reference image:", ref_image_path)
            # For virtual try-on, optionally preprocess the garment (reference) image.
            if control_type == "virtual_tryon" and preprocess_garment:
                if isinstance(ref_image_path, str) and ref_image_path.lower().endswith('.png'):
                    print("Preprocessing garment image...")
                    ref_image = preprocess_garment_image(ref_image_path)
                else:
                    raise ValueError("Reference garment image must be a PNG file when preprocessing is enabled.")
            else:
                ref_image = Image.open(ref_image_path)
            print("Resizing reference image...")
            ref_image = resize_and_center(ref_image, 768, 1024)
        except Exception as e:
            print("Error processing reference image:", e)
            raise

        src_image_array = np.array(src_image)

        try:
            if control_type == "virtual_tryon":
                print("Converting source image to RGB for virtual try-on...")
                src_image = src_image.convert("RGB")
                print("Running human parsing...")
                model_parse, _ = self.parsing(src_image.resize((384, 512)))
                print("Running OpenPose...")
                keypoints = self.openpose(src_image.resize((384, 512)))
                if vt_model_type == "viton_hd":
                    print("Generating garment agnostic mask using VITON-HD...")
                    mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
                # Commented out since DressCode model is not used.
                # elif vt_model_type == "dress_code":
                #     mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
                print("Resizing mask...")
                mask = mask.resize((768, 1024))
            # Commented out pose transfer since it is not used.
            # elif control_type == "pose_transfer":
            #     print("Creating full white mask for pose transfer...")
            #     mask = Image.fromarray(np.ones_like(src_image_array) * 255)
        except Exception as e:
            print("Error during mask generation:", e)
            raise

        try:
            if control_type == "virtual_tryon":
                if vt_model_type == "viton_hd":
                    print("Predicting segmentation for densepose using VITON-HD...")
                    src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                    src_image_seg = Image.fromarray(src_image_seg_array)
                    densepose = src_image_seg
                # Commented out since DressCode model is not used.
                # elif vt_model_type == "dress_code":
                #     src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
                #     src_image_seg_array = src_image_iuv_array[:, :, 0:1]
                #     src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
                #     src_image_seg = Image.fromarray(src_image_seg_array)
                #     densepose = src_image_seg
            # Commented out pose transfer branch.
            # elif control_type == "pose_transfer":
            #     src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)[:, :, ::-1]
            #     src_image_iuv = Image.fromarray(src_image_iuv_array)
            #     densepose = src_image_iuv
            print("Densepose prediction complete.")
        except Exception as e:
            print("Error during densepose prediction:", e)
            raise

        try:
            print("Preparing data for inference...")
            transform = LeffaTransform()
            data = {
                "src_image": [src_image],
                "ref_image": [ref_image],
                "mask": [mask],
                "densepose": [densepose],
            }
            data = transform(data)
            print("Data transformation complete.")
        except Exception as e:
            print("Error during data transformation:", e)
            raise

        try:
            if control_type == "virtual_tryon":
                # Use only the VITON-HD inference.
                print("Selecting VITON-HD inference...")
                inference = self.vt_inference_hd
                # Commented out inference selection for DressCode and Pose Transfer.
                # if vt_model_type == "dress_code":
                #     inference = self.vt_inference_dc
            # elif control_type == "pose_transfer":
            #     inference = self.pt_inference

            print("Running inference...")
            output = inference(
                data,
                ref_acceleration=ref_acceleration,
                num_inference_steps=step,
                guidance_scale=scale,
                seed=seed,
                repaint=vt_repaint,
            )
            print("Inference completed successfully.")
        except Exception as e:
            print("Error during inference:", e)
            raise

        try:
            gen_image = output["generated_image"][0]
            print("Returning generated image, mask, and densepose.")
            return np.array(gen_image), np.array(mask), np.array(densepose)
        except Exception as e:
            print("Error processing inference output:", e)
            raise

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint, preprocess_garment):
        print("Calling leffa_predict_vt with parameters:")
        print(f"src_image_path: {src_image_path}, ref_image_path: {ref_image_path}, vt_model_type: {vt_model_type}")
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            ref_acceleration,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            vt_repaint,
            preprocess_garment,  # Pass through the new flag.
        )

    # Commented out the pose transfer inference function.
    # def leffa_predict_pt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed):
    #     return self.leffa_predict(
    #         src_image_path,
    #         ref_image_path,
    #         "pose_transfer",
    #         ref_acceleration,
    #         step,
    #         scale,
    #         seed,
    #     )

leffa_predictor = LeffaPredictor()

# The main execution block and Gradio UI can also be adapted to remove the pose transfer parts,
# if only VITON-HD (virtual try-on) is required.

# if __name__ == "__main__":
#     print("Starting LeffaPredictor demo...")
#     leffa_predictor = LeffaPredictor()
#     example_dir = "./ckpts/examples"
#     person1_images = list_dir(f"{example_dir}/person1")
#     person2_images = list_dir(f"{example_dir}/person2")
#     garment_images = list_dir(f"{example_dir}/garment")
#
#     title = "## Leffa: Learning Flow Fields in Attention for Controllable Person Image Generation"
#     link = """[📚 Paper](https://arxiv.org/abs/2412.08486) - [🤖 Code](https://github.com/franciszzj/Leffa) - [🔥 Demo](https://huggingface.co/spaces/franciszzj/Leffa) - [🤗 Model](https://huggingface.co/franciszzj/Leffa)
#            
#            Star ⭐ us if you like it!
#            """
#     news = """## News
#             - 09/Jan/2025. Inference defaults to float16, generating an image in 6 seconds (on A100).
#
#             More news can be found in the [GitHub repository](https://github.com/franciszzj/Leffa).
#             """
#     description = "Leffa is a unified framework for controllable person image generation that enables precise manipulation of both appearance (i.e., virtual try-on) and pose (i.e., pose transfer)."
#     note = "Note: The models used in the demo are trained solely on academic datasets. Virtual try-on uses VITON-HD, and pose transfer is disabled."
#
#     with gr.Blocks(theme=gr.themes.Default(primary_hue=gr.themes.colors.pink, secondary_hue=gr.themes.colors.red)).queue() as demo:
#         gr.Markdown(title)
#         gr.Markdown(link)
#         gr.Markdown(news)
#         gr.Markdown(description)
#
#         with gr.Tab("Control Appearance (Virtual Try-on)"):
#             with gr.Row():
#                 with gr.Column():
#                     gr.Markdown("#### Person Image")
#                     vt_src_image = gr.Image(
#                         sources=["upload"],
#                         type="filepath",
#                         label="Person Image",
#                         width=512,
#                         height=512,
#                     )
#                     gr.Examples(
#                         inputs=vt_src_image,
#                         examples_per_page=10,
#                         examples=person1_images,
#                     )
#
#                 with gr.Column():
#                     gr.Markdown("#### Garment Image")
#                     vt_ref_image = gr.Image(
#                         sources=["upload"],
#                         type="filepath",
#                         label="Garment Image",
#                         width=512,
#                         height=512,
#                     )
#                     # New checkbox to choose preprocessing.
#                     preprocess_garment_checkbox = gr.Checkbox(
#                         label="Preprocess Garment Image (PNG only)",
#                         value=False
#                     )
#                     gr.Examples(
#                         inputs=vt_ref_image,
#                         examples_per_page=10,
#                         examples=garment_images,
#                     )
#
#                 with gr.Column():
#                     gr.Markdown("#### Generated Image")
#                     vt_gen_image = gr.Image(
#                         label="Generated Image",
#                         width=512,
#                         height=512,
#                     )
#                     with gr.Row():
#                         vt_gen_button = gr.Button("Generate")
#                     with gr.Accordion("Advanced Options", open=False):
#                         vt_model_type = gr.Radio(
#                             label="Model Type",
#                             choices=[("VITON-HD (Recommended)", "viton_hd")],
#                             value="viton_hd",
#                         )
#                         vt_garment_type = gr.Radio(
#                             label="Garment Type",
#                             choices=[("Upper", "upper_body"),
#                                      ("Lower", "lower_body"),
#                                      ("Dress", "dresses")],
#                             value="upper_body",
#                         )
#                         vt_ref_acceleration = gr.Radio(
#                             label="Accelerate Reference UNet (may slightly reduce performance)",
#                             choices=[("True", True), ("False", False)],
#                             value=False,
#                         )
#                         vt_repaint = gr.Radio(
#                             label="Repaint Mode",
#                             choices=[("True", True), ("False", False)],
#                             value=False,
#                         )
#                         vt_step = gr.Number(
#                             label="Inference Steps", minimum=30, maximum=100, step=1, value=30)
#                         vt_scale = gr.Number(
#                             label="Guidance Scale", minimum=0.1, maximum=5.0, step=0.1, value=2.5)
#                         vt_seed = gr.Number(
#                             label="Random Seed", minimum=-1, maximum=2147483647, step=1, value=42)
#                     with gr.Accordion("Debug", open=False):
#                         vt_mask = gr.Image(
#                             label="Generated Mask",
#                             width=256,
#                             height=256,
#                         )
#                         vt_densepose = gr.Image(
#                             label="Generated DensePose",
#                             width=256,
#                             height=256,
#                         )
#
#                 vt_gen_button.click(
#                     fn=leffa_predictor.leffa_predict_vt,
#                     inputs=[
#                         vt_src_image, vt_ref_image, vt_ref_acceleration,
#                         vt_step, vt_scale, vt_seed, vt_model_type,
#                         vt_garment_type, vt_repaint, preprocess_garment_checkbox
#                     ],
#                     outputs=[vt_gen_image, vt_mask, vt_densepose]
#                 )
#
#         # Removed the "Control Pose (Pose Transfer)" tab as pose transfer is not used.
#
#         gr.Markdown(note)
#         demo.launch(share=True, server_port=7860, allowed_paths=["./ckpts/examples"])
