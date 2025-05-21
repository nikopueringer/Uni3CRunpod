import argparse

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from src.dataset import load_dataset
from src.models.pcd_controller import PCDController
from src.pipelines.pipeline_pcd import PCDControllerPipeline

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", default=None, type=str, required=True, help="the path of input image")
    parser.add_argument("--render_path", default=None, type=str, required=True, help="the path of render folder")
    parser.add_argument("--output_path", default="result.mp4", type=str, help="output path")
    parser.add_argument("--nframe", default=81, type=int, help="Total number of frames")
    parser.add_argument("--prompt", default="This video describes a slow and stable camera movement with high quality and high definition.",
                        type=str, help="Prompt of the reference image")
    parser.add_argument("--max_area", default=480 * 768, type=int, help="Total pixel area of height * width")
    args = parser.parse_args()

    cfg = OmegaConf.load(hf_hub_download(repo_id="ewrfcas/Uni3C", filename="config.json"))
    base_model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    print("loading transformer...")
    transformer = PCDController.from_pretrained(base_model_id, subfolder="transformer", controlnet_cfg=cfg.controlnet_cfg, torch_dtype=torch.bfloat16)
    print("loading controlnet...")
    transformer.build_controlnet(model_path="controlnet.pth")
    pipe = PCDControllerPipeline(
        tokenizer=AutoTokenizer.from_pretrained(base_model_id, subfolder="tokenizer"),
        text_encoder=UMT5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16),
        image_encoder=CLIPVisionModel.from_pretrained(base_model_id, subfolder="image_encoder", torch_dtype=torch.float32),
        image_processor=CLIPImageProcessor.from_pretrained(base_model_id, subfolder="image_processor"),
        transformer=transformer,
        vae=AutoencoderKLWan.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_id, subfolder="scheduler")
    )
    device = "cuda"

    # replace this with pipe.to("cuda") if you have sufficient VRAM
    pipe.enable_model_cpu_offload()

    image, render_video, render_mask, camera_embedding, height, width = load_dataset(
        reference_image=args.reference_image,
        render_path=args.render_path,
        nframe=args.nframe,
        max_area=args.max_area,
        pipe=pipe,
        use_camera_embedding=cfg.get("camera_embedding", False),
        device=device
    )

    output = pipe(
        image=image,
        render_video=render_video.to(device),
        render_mask=render_mask.to(device),
        camera_embedding=camera_embedding.to(device),
        prompt=(args.prompt),
        negative_prompt="",
        height=height,
        width=width,
        num_frames=args.nframe,
        guidance_scale=5.0,
    ).frames[0]
    export_to_video(output, args.output_path, fps=16)
