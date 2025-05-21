import argparse
import json
import os
import warnings

import einops
import numpy as np
import torch
import trimesh
from PIL import Image, ImageOps
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from diffusers.utils import export_to_video
from pytorch3d.renderer import PointsRasterizationSettings
from torchvision.transforms import ToTensor, ToPILImage

import third_party.depth_pro.depth_pro as depth_pro
from src.pointcloud import point_rendering
from src.utils import traj_map, points_padding, np_points_padding, set_initial_camera, build_cameras
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")


def main():
    torch.set_grad_enabled(False)
    # == parse configs ==
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_image", default=None, type=str, required=True,
                        help="the path of input image")
    parser.add_argument("--output_path", default="outputs/temp", type=str,
                        help="output folder's path")
    parser.add_argument("--traj_type", default="custom", type=str,
                        choices=["custom", "free1", "free2", "free3", "free4", "free5", "swing1", "swing2", "orbit"],
                        help="custom refers to a custom trajectory, while the others are pre-defined camera trajectories (see traj_map for details)")
    parser.add_argument("--nframe", default=81, type=int, help="Total number of frames")
    parser.add_argument("--d_r", default=1.0, type=float,
                        help="Camera distance, default is 1.0, range 0.25 to 2.5")
    parser.add_argument("--d_theta", default=0.0, type=float,
                        help="Vertical rotation, <0 up, >0 down, range -90 to 30; generally not recommended to angle too much downwards")
    parser.add_argument("--d_phi", default=0.0, type=float,
                        help="Horizontal rotation, <0 right, >0 left, supports 360 degrees; range -360 to 360")
    parser.add_argument("--x_offset", default=0.0, type=float,
                        help="Horizontal translation, <0 left, >0 right, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--y_offset", default=0.0, type=float,
                        help="Vertical translation, <0 up, >0 down, range -0.5 to 0.5; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--z_offset", default=0.0, type=float,
                        help="Forward and backward translation, <0 back, >0 forward, range -0.5 to 0.5 is ok; depends on depth, excessive movement may cause artifacts")
    parser.add_argument("--focal_length", default=1.0, type=float,
                        help="Focal length, range 0.25 to 2.5; changing focal length zooms in and out")
    parser.add_argument("--start_elevation", default=5.0, type=float,
                        help="Initial angle, no exceptions to change")

    args = parser.parse_args()
    device = "cuda"
    print("Init depth model")
    depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
    depth_model = depth_model.eval()

    print("Init mask model")
    seg_net = TracerUniversalB7(device=device, batch_size=1,
                                model_path=hf_hub_download(repo_id="ewrfcas/Uni3C",
                                                           filename="tracer_b7.pth",
                                                           repo_type="model")).eval()
    # seg_net = TracerUniversalB7(device=device, batch_size=1, model_path="./check_points/tracer_b7.pth").eval()

    # == motion definition ==
    if args.traj_type == "custom":
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = \
            "free", args.x_offset, args.y_offset, args.z_offset, args.d_theta, args.d_phi, args.d_r
    else:
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(args.traj_type)

    # load image
    image = Image.open(args.reference_image).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w_origin, h_origin = image.size
    hw_list = [[480, 768], [512, 720], [608, 608], [720, 512], [768, 480]]
    hw_ratio_list = [h_ / w_ for [h_, w_] in hw_list]
    hw_ratio = h_origin / w_origin
    sub_hw_ratio = np.abs(np.array(hw_ratio_list) - hw_ratio)
    select_idx = np.argmin(sub_hw_ratio)
    height, width = hw_list[select_idx]
    print(f"Image: {args.reference_image.split('/')[-1]}, Resolution: {h_origin}x{w_origin}->{height}x{width}")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1

    os.makedirs(args.output_path, exist_ok=True)

    # inference depth
    with torch.no_grad():
        depth_image = np.array(image)
        depth_image = depth_transform(depth_image)
        prediction = depth_model.infer(depth_image, f_px=None)
        depth = prediction["depth"]  # Depth in [m].
        depth = depth[None, None]
        focallength_px = prediction["focallength_px"].item()  # Focal length in pixels.
        K = torch.tensor([[focallength_px, 0, width / 2],
                          [0, focallength_px, height / 2],
                          [0, 0, 1]], dtype=torch.float32)
        K_inv = K.inverse()
        intrinsic = K[None].repeat(args.nframe, 1, 1)

    # get pointcloud
    points2d = torch.stack(torch.meshgrid(torch.arange(width, dtype=torch.float32),
                                          torch.arange(height, dtype=torch.float32), indexing="xy"), -1)  # [h,w,2]
    points3d = points_padding(points2d).reshape(height * width, 3)  # [hw,3]
    points3d = (K_inv @ points3d.T * depth.reshape(1, height * width).cpu()).T
    colors = ((depth_image + 1) / 2 * 255).to(torch.uint8).permute(1, 2, 0).reshape(height * width, 3)
    points3d = points3d.cpu().numpy()
    colors = colors.cpu().numpy()

    # inference foreground mask
    with torch.no_grad():
        origin_w_, origin_h_ = image.size
        image_pil = image.resize((512, 512))
        fg_mask = seg_net([image_pil])[0]
        fg_mask = fg_mask.resize((origin_w_, origin_h_))
    fg_mask = np.array(fg_mask)
    fg_mask = fg_mask > 127.5
    fg_mask = torch.tensor(fg_mask)
    if fg_mask.float().mean() < 0.05:
        fg_mask[...] = True
    depth_avg = torch.median(depth[0, 0, fg_mask]).item()

    w2c_0, c2w_0 = set_initial_camera(args.start_elevation, depth_avg)

    # convert points3d to the world coordinate
    points3d = (c2w_0.numpy()[:3] @ np_points_padding(points3d).T).T
    pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    _ = pcd.export(f"{args.output_path}/pcd.ply")

    # build camera viewpoints according to d_theta，d_phi, d_r
    w2cs, c2ws, intrinsic = build_cameras(cam_traj=cam_traj,
                                          w2c_0=w2c_0,
                                          c2w_0=c2w_0,
                                          intrinsic=intrinsic,
                                          nframe=args.nframe,
                                          focal_length=args.focal_length,
                                          d_theta=d_theta,
                                          d_phi=d_phi,
                                          d_r=d_r,
                                          radius=depth_avg,
                                          x_offset=x_offset,
                                          y_offset=y_offset,
                                          z_offset=z_offset)

    # save camera infos
    w2cs_list = w2cs.cpu().numpy().tolist()
    camera_infos = {"intrinsic": K.cpu().numpy().tolist(), "extrinsic": w2cs_list, "height": height, "width": width}
    with open(f"{args.output_path}/cam_info.json", "w") as writer:
        json.dump(camera_infos, writer, indent=2)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        control_imgs, render_masks = point_rendering(K=intrinsic.float(),
                                                     w2cs=w2cs.float(),
                                                     depth=depth.float(),
                                                     image=validation_image.float() * 2 - 1,
                                                     raster_settings=PointsRasterizationSettings(image_size=(height, width),
                                                                                                 radius=0.008,
                                                                                                 points_per_pixel=8),
                                                     device=device,
                                                     background_color=[0, 0, 0],
                                                     sobel_threshold=0.35,
                                                     sam_mask=None)

    control_imgs = einops.rearrange(control_imgs, "(b f) c h w -> b c f h w", f=args.nframe)
    render_masks = einops.rearrange(render_masks, "(b f) c h w -> b c f h w", f=args.nframe)

    render_video = []
    mask_video = []
    control_imgs = control_imgs.to(torch.float32)
    for i in range(args.nframe):
        img = ToPILImage()((control_imgs[0][:, i] + 1) / 2)
        render_video.append(img)
        mask = ToPILImage()(render_masks[0][:, i])
        mask_video.append(mask)

    export_to_video(render_video, f"{args.output_path}/render.mp4", fps=16)
    export_to_video(mask_video, f"{args.output_path}/render_mask.mp4", fps=16)

    print("Rendering finished.")


if __name__ == "__main__":
    main()
