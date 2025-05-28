import argparse
import json
import os.path
import pickle

import cv2
import numpy as np
import roma
import torch
import trimesh
from PIL import Image, ImageOps
from diffusers.utils import export_to_video
from geocalib import GeoCalib
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PerspectiveCameras,
    MeshRasterizer,
    HardPhongShader,
    Materials,
    RasterizationSettings,
    BlendParams,
    TexturesVertex,
)
from pytorch3d.structures import Pointclouds, Meshes
from torchvision.transforms import ToTensor
from ultralytics import YOLO

import third_party.depth_pro.depth_pro as depth_pro
from src.pointcloud import PointsZbufRenderer, get_boundaries_mask, suppress_stdout_stderr
from src.render import MeshRendererWrap, create_lights
from src.utils import points_padding, np_points_padding, rotation_matrix_from_vectors, traj_map, set_initial_camera

import decord


def build_trajectory(nframe, radius, c2w_0, w2c_0, d_r, x_offset, y_offset,
                     d_theta, d_phi, vertices, smpl_follow=False):
    c2ws = [c2w_0]
    w2cs = [w2c_0]
    d_thetas, d_phis, d_rs = [], [], []
    x_offsets, y_offsets = [], []

    vertices = vertices.mean(axis=1)  # [N,3]
    smpl_movement = vertices[1:] - vertices[:-1]  # [N-1,3]
    smpl_movement = torch.from_numpy(np.concatenate([np.zeros((1, 3)), smpl_movement], axis=0)).to(torch.float32)  # [N,3]
    smpl_movement = torch.cumsum(smpl_movement, dim=0)

    for i in range(nframe - 1):
        coef = (i + 1) / (nframe - 1)
        d_thetas.append(d_theta * coef)
        d_phis.append(d_phi * coef)
        d_rs.append(coef * d_r + (1 - coef) * 1.0)
        x_offsets.append(radius * x_offset * ((i + 1) / nframe))
        y_offsets.append(radius * y_offset * ((i + 1) / nframe))

    for i in range(nframe - 1):
        d_theta_rad = np.deg2rad(d_thetas[i])
        R_theta = torch.tensor([[1, 0, 0, 0],
                                [0, np.cos(d_theta_rad), -np.sin(d_theta_rad), 0],
                                [0, np.sin(d_theta_rad), np.cos(d_theta_rad), 0],
                                [0, 0, 0, 1]], dtype=torch.float32)
        d_phi_rad = np.deg2rad(d_phis[i])
        R_phi = torch.tensor([[np.cos(d_phi_rad), 0, np.sin(d_phi_rad), 0],
                              [0, 1, 0, 0],
                              [-np.sin(d_phi_rad), 0, np.cos(d_phi_rad), 0],
                              [0, 0, 0, 1]], dtype=torch.float32)
        c2w_1 = R_phi @ R_theta @ c2w_0
        if i < len(x_offsets) and i < len(y_offsets):
            c2w_1[:3, -1] += torch.tensor([x_offsets[i], y_offsets[i], 0])
        if smpl_follow:
            c2w_1[:3, -1] += smpl_movement[i + 1]
        c2w_1[:3, -1] *= d_rs[i]
        w2c_1 = c2w_1.inverse()
        c2ws.append(c2w_1)
        w2cs.append(w2c_1)

    return c2ws, w2cs


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default=None, type=str, required=True, help="smpl dir")
    parser.add_argument("--input_image", default=None, type=str, required=True, help="reference image dir")
    parser.add_argument("--output_path", default="outputs/uni3c_temp", type=str, help="output path")
    parser.add_argument("--traj_type", default="custom", type=str,
                        choices=["custom", "free1", "free2", "free3", "free4", "free5", "swing1", "swing2", "orbit"],
                        help="custom refers to a custom trajectory, while the others are pre-defined camera trajectories (see traj_map for details)")
    parser.add_argument("--output_video_length", default=5.0, type=float, help="output video length (sec)")
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
    parser.add_argument("--start_elevation", default=5.0, type=float,
                        help="Initial angle, no exceptions to change")
    parser.add_argument("--input_fps", default=None, type=int,
                        help="the fps of the original video, keep it None if we have cs_map.mp4")
    parser.add_argument("--smpl_follow", action="store_true", help="whether the camera will move along the human body")

    args = parser.parse_args()

    # 固定路径和setting
    smplx_sc_color_path = "./third_party/GVHMR_realisdance/hmr4d/utils/body_model/smplx_color.pt"
    smplx_face_path = "./third_party/GVHMR_realisdance/hmr4d/utils/body_model/faces.npy"
    mano_idx_path = "./third_party/GVHMR_realisdance/hmr4d/utils/body_model/MANO_SMPLX_vertex_ids.pkl"  # hand smpl index
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))

    device = "cuda"
    print("Init depth model")
    depth_model, depth_transform = depth_pro.create_model_and_transforms(device=device)
    depth_model = depth_model.eval()

    print("Init GeoCalib")
    geo_model = GeoCalib().to(device)

    print("Init Yolo")
    yolo_model = YOLO("yolo11n-pose.pt")

    # == motion definition ==
    if args.traj_type == "custom":
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = \
            "free", args.x_offset, args.y_offset, args.z_offset, args.d_theta, args.d_phi, args.d_r
    else:
        cam_traj, x_offset, y_offset, z_offset, d_theta, d_phi, d_r = traj_map(args.traj_type)

    # set data_path
    key_points3d_smpl_file = f"{args.input_path}/coco_17joints.npy"
    smplx_full_vertices_path = f"{args.input_path}/full_vertices.npy"
    os.makedirs(args.output_path, exist_ok=True)

    # load image
    image = Image.open(args.input_image).convert("RGB")
    image = ImageOps.exif_transpose(image)
    w_origin, h_origin = image.size
    hw_list = [[480, 768], [512, 720], [608, 608], [720, 512], [768, 480]]
    hw_ratio_list = [h_ / w_ for [h_, w_] in hw_list]
    hw_ratio = h_origin / w_origin
    sub_hw_ratio = np.abs(np.array(hw_ratio_list) - hw_ratio)
    select_idx = np.argmin(sub_hw_ratio)
    height, width = hw_list[select_idx]
    print(f"Image: {args.input_path.split('/')[-1]}, Resolution: {h_origin}x{w_origin}->{height}x{width}")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    validation_image = ToTensor()(image)[None]  # [1,c,h,w], 0~1

    raster_settings = PointsRasterizationSettings(
        image_size=(height, width),
        radius=0.008,
        points_per_pixel=8
    )
    mesh_raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )

    # inference depth
    depth_image = np.array(image)
    depth_image = depth_transform(depth_image)
    prediction = depth_model.infer(depth_image, f_px=None)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"].item()  # Focal length in pixels.

    # gravity calibration
    geo_inputs = geo_model.load_image(args.input_image).to(device)
    focal_length_tensor = torch.tensor([focallength_px / width * w_origin, focallength_px / height * h_origin],
                                       dtype=torch.float32).mean().to(device)
    result = geo_model.calibrate(geo_inputs, priors={"focal": focal_length_tensor})
    c_gravity = result["gravity"]._data.cpu().T  # [1,3]->[3,1]

    K = torch.tensor([[focallength_px, 0, width / 2],
                      [0, focallength_px, height / 2],
                      [0, 0, 1]], dtype=torch.float32)
    K_inv = K.inverse()
    depth = torch.clip(depth, 1e-4, 100)  # clip for inf depth

    # unproject into 3D points
    points2d = torch.stack(torch.meshgrid(torch.arange(width, dtype=torch.float32),
                                          torch.arange(height, dtype=torch.float32), indexing="xy"), -1)  # [h,w,2]
    points3d = points_padding(points2d).reshape(height * width, 3)  # [hw,3]
    points3d = (K_inv @ points3d.T * depth.reshape(1, height * width).cpu()).T
    points3d = points3d.cpu().numpy()

    # detect key point 2d
    yolo_result = yolo_model(args.input_image)[0]
    kpts = yolo_result.keypoints.data
    xyxy = yolo_result.boxes.xyxy
    mean_conf = kpts[:, :, -1].mean(dim=-1)
    if xyxy.shape[0] == 0 or torch.sum(mean_conf > 0.5) < 1:
        raise ValueError("No people is detected in the image")
    kpts = kpts[mean_conf > 0.5]
    xyxy = xyxy[mean_conf > 0.5]
    yolo_areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    max_area_idx = torch.argmax(yolo_areas)
    key_point2d = kpts[max_area_idx][None].cpu()  # [1,17,3]

    # get key_point2d from reference image
    key_prob = key_point2d[0, :, 2]
    key_prob[key_prob < 0.7] = 0

    key_point2d = key_point2d[0, :, :2]
    key_point2d[:, 0] = key_point2d[:, 0] * (width / w_origin)
    key_point2d[:, 1] = key_point2d[:, 1] * (height / h_origin)
    key_point2d = np.round(key_point2d.numpy()).astype(np.int32)
    key_point2d[:, 0] = np.clip(key_point2d[:, 0], 0, width - 1)
    key_point2d[:, 1] = np.clip(key_point2d[:, 1], 0, height - 1)

    key_3d_index = key_point2d[:, 1] * width + key_point2d[:, 0]
    # 剔除depth异常点
    depth_avg = depth.reshape(-1)[key_3d_index].median().item()
    valid_key_depth = (depth.reshape(-1)[key_3d_index]) < depth_avg * 3
    key_prob[~valid_key_depth] = 0

    assert (key_prob > 0).sum() >= 3, "2D pose of the reference image is invalid"

    # draw key points
    image_np = np.array(image)
    # draw pointed image
    for point in key_point2d:
        x, y = int(point[0]), int(point[1])
        cv2.circle(image_np, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    image_pointed = Image.fromarray(image_np)
    image_pointed.save(f"{args.output_path}/image_pointed.png")

    # get the initial extrinsic
    w2c_0, c2w_0 = set_initial_camera(args.start_elevation, depth_avg)

    w_gravity = c2w_0[:3, :3] @ c_gravity  # [3,1]
    w_gravity = w_gravity.T  # [1,3]
    w_gravity = w_gravity / torch.norm(w_gravity, dim=1, keepdim=True)

    # 获取世界坐标系下的depth-pro环境点云
    points3d = (c2w_0.numpy()[:3] @ np_points_padding(points3d).T).T
    colors = np.array(image).reshape(height * width, 3)

    # save environment pointcloud
    env_pcd = trimesh.PointCloud(vertices=points3d, colors=colors)
    env_pcd.export(f"{args.output_path}/env_pcd.ply")

    # get 17 key points in the depth-pro's 3d scene
    key_point3d = points3d[key_3d_index]

    # align key_point3d and key_smpl_point3d
    key_smpl_point3d = np.load(key_points3d_smpl_file)  # [17,3]
    if key_smpl_point3d.ndim == 3:
        key_smpl_point3d = key_smpl_point3d[0]

    R, T, s = roma.rigid_points_registration(torch.tensor(key_smpl_point3d), torch.tensor(key_point3d),
                                             weights=key_prob, compute_scaling=True)  # weights=None

    # load smpl data
    smplx_color = torch.load(smplx_sc_color_path, weights_only=False)
    smplx_cs_colors = torch.from_numpy(smplx_color).unsqueeze(0) / 255.0
    vertices = torch.from_numpy(np.load(smplx_full_vertices_path))
    faces = torch.from_numpy(np.load(smplx_face_path)).unsqueeze(0)

    # set target video fps
    new_fps = 16
    ori_video_length = vertices.shape[0]
    # to get the fps of the original video
    if args.input_fps is None:
        meta_reader = decord.VideoReader(f"{args.input_path}/cs_map.mp4")
        ori_fps = meta_reader.get_avg_fps()
    else:
        ori_fps = args.input_fps
    nframe = int(args.output_video_length * new_fps) // 4 * 4 + 1  # final output frames
    # sampling 'new_fps / ori_fps' frames，ensuring the output video is fps=16
    normed_video_length = max(round(ori_video_length / ori_fps * new_fps), nframe)
    normed_video_length = int(normed_video_length) // 4 * 4 + 1
    print("video frames:", ori_video_length, "normed to:", normed_video_length, "clip to:", nframe)
    select_idx = np.linspace(0, ori_video_length - 1, normed_video_length).round().astype(int)[:nframe]
    vertices = vertices[select_idx]

    textures = TexturesVertex(verts_features=smplx_cs_colors.repeat(nframe, 1, 1).to(device))

    # the alignment operation
    vertices = (R[None] @ (vertices * s).mT).mT + T[None, None]

    # smpl's ground always point to y
    smpl_gravity = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).reshape(3, 1)
    smpl_gravity = R @ smpl_gravity  # [3,1]
    smpl_gravity = smpl_gravity.numpy().reshape(-1)
    w_gravity = w_gravity.numpy().reshape(-1)
    calibrate_R = torch.from_numpy(rotation_matrix_from_vectors(smpl_gravity, w_gravity))[None].to(torch.float32)
    vertices = (calibrate_R @ vertices.mT).mT

    # render new camera views
    c2ws, w2cs = build_trajectory(nframe=nframe, radius=depth_avg, c2w_0=c2w_0, w2c_0=w2c_0, d_r=d_r,
                                  x_offset=x_offset, y_offset=y_offset, d_theta=d_theta, d_phi=d_phi,
                                  vertices=vertices, smpl_follow=args.smpl_follow)
    w2cs = torch.stack(w2cs)  # opencv coordinate
    c2ws = torch.stack(c2ws)

    # save camera infos
    meta_info = {
        "d_r": d_r,
        "x_offset": x_offset,
        "y_offset": y_offset,
        "d_phi": d_phi,
        "d_theta": d_theta,
        "smpl_x_offset": 0,
        "smpl_y_offset": 0,
        "smpl_z_offset": 0,
        "smpl_follow": args.smpl_follow,
        "use_geocalib": True
    }
    cam_info = {"intrinsic": K.numpy().tolist(), "extrinsic": w2cs.numpy().tolist(), "height": height, "width": width, "meta_info": meta_info}
    with open(f"{args.output_path}/cam_info.json", "w") as writer:
        json.dump(cam_info, writer)

    # 获取手部vertices
    idxs_data = pickle.load(open(mano_idx_path, 'rb'))
    hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']])
    left_hand_idxs = idxs_data['left_hand']
    right_hand_idxs = idxs_data['right_hand']
    left_hand_colors = np.tile(np.array([0.5, 1.0, 0.5]).reshape((1, 1, 3)), (1, idxs_data['left_hand'].shape[0], 1))
    right_hand_colros = np.tile(np.array([1.0, 0.5, 0.5]).reshape((1, 1, 3)), (1, idxs_data['right_hand'].shape[0], 1))
    hand_colors = torch.from_numpy(np.concatenate([left_hand_colors, right_hand_colros], axis=1)).to(dtype=torch.float32)
    hand_textures = TexturesVertex(verts_features=hand_colors.repeat(nframe, 1, 1).to(device))

    # load hand vertices from hamer_info.pkl
    if os.path.exists(f"{args.input_path}/hamer_info.pkl"):
        hand_info = pickle.load(open(f"{args.input_path}/hamer_info.pkl", "rb"))
        left_hand_vertices = []
        right_hand_vertices = []
        valid_hand = []
        for iframe in select_idx:
            if hand_info[iframe] is None or len(hand_info[iframe]['all_right']) != 2:  # no hands are detected，fail
                left_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                right_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                valid_hand.append(False)
            elif hand_info[iframe]['all_right'][0] + hand_info[iframe]['all_right'][1] != 1:  # two left or right hands are detected，fail
                left_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                right_hand_vertices.append(np.zeros((778, 3), dtype=np.float32))
                valid_hand.append(False)
            else:
                for j in range(len(hand_info[iframe]['all_right'])):
                    is_right = hand_info[iframe]['all_right'][j].item()
                    if is_right == 1:
                        right_hand_vertices.append(hand_info[iframe]['all_verts'][j])
                    else:
                        left_hand_vertices.append(hand_info[iframe]['all_verts'][j])
                valid_hand.append(True)

        smpl_left_hand_vertices = vertices[:, left_hand_idxs].to(dtype=torch.float32)
        smpl_right_hand_vertices = vertices[:, right_hand_idxs].to(dtype=torch.float32)

        left_hand_vertices = torch.tensor(np.array(left_hand_vertices), dtype=torch.float32)
        right_hand_vertices = torch.tensor(np.array(right_hand_vertices), dtype=torch.float32)
        for j in range(len(valid_hand)):
            if valid_hand[j]:
                R1, T1, s1 = roma.rigid_points_registration(left_hand_vertices[j],
                                                            smpl_left_hand_vertices[j],
                                                            compute_scaling=True)  # weights=None
                left_hand_vertices[j] = (R1 @ (left_hand_vertices[j] * s1).mT).mT + T1[None]
                R2, T2, s2 = roma.rigid_points_registration(right_hand_vertices[j],
                                                            smpl_right_hand_vertices[j],
                                                            compute_scaling=True)  # weights=None
                right_hand_vertices[j] = (R2 @ (right_hand_vertices[j] * s2).mT).mT + T2[None]
        valid_hand = torch.tensor(valid_hand).to(torch.float32)[:, None, None]  # [F,1,1]
        left_hand_vertices = left_hand_vertices * valid_hand + smpl_left_hand_vertices * (1 - valid_hand)
        right_hand_vertices = right_hand_vertices * valid_hand + smpl_right_hand_vertices * (1 - valid_hand)
        hand_vertices = torch.cat([left_hand_vertices, right_hand_vertices], dim=1)

    else:  # no hamer inputs, directly copy smpl hands
        # get hand vertices
        valid_hand = None
        idxs_data = pickle.load(open(mano_idx_path, 'rb'))
        hand_idxs = np.concatenate([idxs_data['left_hand'], idxs_data['right_hand']])
        left_hand_colors = np.tile(np.array([0.5, 1.0, 0.5]).reshape((1, 1, 3)), (1, idxs_data['left_hand'].shape[0], 1))
        right_hand_colros = np.tile(np.array([1.0, 0.5, 0.5]).reshape((1, 1, 3)), (1, idxs_data['right_hand'].shape[0], 1))
        hand_colors = torch.from_numpy(np.concatenate([left_hand_colors, right_hand_colros], axis=1)).to(dtype=torch.float32)
        hand_textures = TexturesVertex(verts_features=hand_colors.repeat(nframe, 1, 1).to(device))
        hand_vertices = vertices[:, hand_idxs].to(dtype=torch.float32)

    # hash mapping
    hand_set = set(hand_idxs.tolist())
    old_to_new = {old: new for new, old in enumerate(hand_idxs)}

    hand_faces = []
    for face in faces[0]:
        v0, v1, v2 = face
        v0, v1, v2 = v0.item(), v1.item(), v2.item()
        if all(v in hand_set for v in [v0, v1, v2]):
            new_face = [old_to_new[v0], old_to_new[v1], old_to_new[v2]]
            hand_faces.append(new_face)

    hand_faces = np.array(hand_faces, dtype=np.int64)[None]  # [1, Y, 3]
    # stitching mesh faces from hands
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279], [122, 118, 279],
                          [279, 118, 215], [118, 117, 215], [215, 117, 214], [117, 119, 214], [214, 119, 121],
                          [119, 120, 121], [121, 120, 78], [120, 108, 78], [78, 108, 79]]).reshape((1, -1, 3))
    hand_faces = torch.from_numpy(np.concatenate([hand_faces, faces_new], axis=1)).to(dtype=torch.long)

    # post-processing for pointcloud
    points3d = torch.tensor(points3d)
    colors = torch.tensor(colors)
    disp = 1 / (depth + 1e-7)
    boundary_mask = get_boundaries_mask(disp[None, None], sobel_threshold=0.35)[0, 0]
    boundary_mask = boundary_mask.reshape(-1).cpu()
    points3d = points3d[boundary_mask == False]
    colors = colors[boundary_mask == False]
    colors = (colors / 255).to(torch.float32)
    point_cloud = Pointclouds(points=[points3d.to(device)], features=[colors.to(device)]).extend(nframe)

    # convert opencv to opengl coordinate
    c2ws[:, :, 0] = - c2ws[:, :, 0]
    c2ws[:, :, 1] = - c2ws[:, :, 1]
    w2cs = c2ws.inverse()
    K = K[None]
    focal_length = torch.stack([K[:, 0, 0], K[:, 1, 1]], dim=1)
    principal_point = torch.stack([K[:, 0, 2], K[:, 1, 2]], dim=1)
    image_shapes = torch.tensor([[height, width]]).repeat(nframe, 1)
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point,
                                 R=c2ws[:, :3, :3], T=w2cs[:, :3, -1], in_ndc=False,
                                 image_size=image_shapes, device=device)
    renderer = PointsZbufRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=[0, 0, 0])
    )

    # rendering environment pointclouds into rgb video
    with suppress_stdout_stderr():
        render_rgbs, zbuf = renderer(point_cloud)  # render_rgbs [f,h,w,3]
    render_rgbs[0:1] = torch.tensor(np.array(image) / 255)  # replace the first frame with reference image
    render_masks = (zbuf[..., 0:1] == -1).float()  # [f,h,w,1]

    meshes = Meshes(verts=vertices.to(device), faces=faces.repeat(nframe, 1, 1).to(device), textures=textures)
    materials = Materials(device=device, shininess=0)

    # save & visualize human meshes
    verts_ = meshes.verts_packed().cpu().numpy()
    faces_ = meshes.faces_packed().cpu().numpy()
    tri_mesh = trimesh.Trimesh(vertices=verts_, faces=faces_)
    tri_mesh.export(f"{args.output_path}/smpl_meshes.ply")

    mesh_renderer = MeshRendererWrap(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=mesh_raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params)
    )

    with suppress_stdout_stderr():
        render_mesh_rgbs, fragments = mesh_renderer(meshes, cameras=cameras, materials=materials,
                                                    lights=create_lights(device, c2ws[:, :3, :3]))
    smpl_depth = fragments.zbuf.min(dim=-1)[0]  # [F,H,W]
    render_mesh_rgbs = torch.clip(render_mesh_rgbs, 0, 1)

    # render hands
    hand_materials = Materials(device=device, shininess=0)
    hand_meshes = Meshes(verts=hand_vertices.to(device), faces=hand_faces.repeat(nframe, 1, 1).to(device), textures=hand_textures)
    with suppress_stdout_stderr():
        render_hand_rgbs, fragments = mesh_renderer(hand_meshes, cameras=cameras, materials=hand_materials,
                                                    lights=create_lights(device, c2ws[:, :3, :3], ambient_color=0.3, diffuse_color=0.75, specular_color=0.2))
    hand_depth = fragments.zbuf.min(dim=-1)[0]  # [F,H,W]
    render_hand_rgbs = torch.clip(render_hand_rgbs, 0, 1)  # [F,H,W,4]
    # mask the occluded hands
    hand_mask = (hand_depth <= smpl_depth).float().unsqueeze(-1)  # [F,H,W,1]
    if valid_hand is not None:
        valid_hand = valid_hand.unsqueeze(-1).to(hand_mask.device)  # [F,1,1]->[F,1,1,1]
        hand_mask = hand_mask * (1 - valid_hand) + torch.ones_like(hand_mask) * valid_hand
    render_hand_rgbs_refined = render_hand_rgbs * hand_mask + torch.zeros_like(render_hand_rgbs) * (1 - hand_mask)

    video = []
    env_video = []
    smpl_video = []
    hand_refined_video = []
    mask_video = []
    for i in range(render_rgbs.shape[0]):
        rgb = (render_rgbs[i] * 255).cpu().numpy().astype(np.uint8)
        smpl = (render_mesh_rgbs[i, :, :, :3] * 255).cpu().numpy().astype(np.uint8)
        hand_refined = (render_hand_rgbs_refined[i, :, :, :3] * 255).cpu().numpy().astype(np.uint8)
        mask = render_mesh_rgbs[i, :, :, 3:4].cpu().numpy()
        frame = (rgb * (1 - mask) + smpl * mask).astype(np.uint8)
        render_mask = (render_masks[i, :, :, 0] * 255).cpu().numpy().astype(np.uint8)

        video.append(Image.fromarray(frame))
        env_video.append(Image.fromarray(rgb))
        smpl_video.append(Image.fromarray(smpl))
        hand_refined_video.append(Image.fromarray(hand_refined))
        mask_video.append(Image.fromarray(render_mask))

    export_to_video(video, output_video_path=f"{args.output_path}/combined_render.mp4", fps=new_fps)
    export_to_video(env_video, output_video_path=f"{args.output_path}/env_render.mp4", fps=new_fps)
    export_to_video(smpl_video, output_video_path=f"{args.output_path}/smpl_render.mp4", fps=new_fps)
    export_to_video(hand_refined_video, output_video_path=f"{args.output_path}/hand_render.mp4", fps=new_fps)
    export_to_video(mask_video, output_video_path=f"{args.output_path}/mask_render.mp4", fps=new_fps)
