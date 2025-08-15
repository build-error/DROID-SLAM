import sys
sys.path.append("droid_slam")

import torch
import argparse
import droid_backends
import open3d as o3d
from visualization import create_camera_actor
from lietorch import SE3
from cuda_timer import CudaTimer
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def save_trajectory_tum(poses, filepath, timestamps=None):
    """
    Save trajectory in TUM format for use with evo.
    Format: timestamp tx ty tz qx qy qz qw
    """
    if poses is None or len(poses) == 0:
        print("[INFO] No trajectory to save.")
        return

    poses = np.array(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Poses must have shape (N, 4, 4)")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if timestamps is None:
        timestamps = np.arange(len(poses), dtype=float)
    else:
        timestamps = np.array(timestamps, dtype=float)
        if len(timestamps) != len(poses):
            raise ValueError("Number of timestamps must match number of poses.")

    traj_list = []
    for ts, pose in zip(timestamps, poses):
        t = pose[:3, 3]
        q = R.from_matrix(pose[:3, :3]).as_quat()
        q /= np.linalg.norm(q)  # normalize quaternion
        traj_list.append([ts, t[0], t[1], t[2], q[0], q[1], q[2], q[3]])

    np.savetxt(filepath, traj_list, fmt="%.6f")
    print(f"[INFO] Saved TUM trajectory to: {filepath}")


def view_reconstruction(filename: str, filter_thresh=0.005, filter_count=2, save_ply=None, save_traj=None):
    reconstruction_blob = torch.load(filename)
    images = reconstruction_blob["images"].cuda()[..., ::2, ::2]
    disps = reconstruction_blob["disps"].cuda()[..., ::2, ::2]
    poses = reconstruction_blob["poses"].cuda()
    intrinsics = 4 * reconstruction_blob["intrinsics"].cuda()

    disps = disps.contiguous()
    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:, [2, 1, 0]].permute(0, 2, 3, 1) / 255.0

    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    # Save PLY file if path provided
    if save_ply:
        os.makedirs(os.path.dirname(save_ply), exist_ok=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)
        o3d.io.write_point_cloud(save_ply, pcd)
        print(f"[INFO] Saved point cloud to {save_ply}")

    # Get pose matrices as Nx4x4 numpy array
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()

    # Save trajectory if path provided
    if save_traj:
        save_trajectory_tum(pose_mats, save_traj)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=960, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)
    vis.add_geometry(point_cloud)

    for i in range(len(poses)):
        cam_actor = create_camera_actor(False)
        cam_actor.transform(pose_mats[i])
        vis.add_geometry(cam_actor)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="path to .pth reconstruction file")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--save_ply", type=str, help="Path to save PLY file", default=None)
    parser.add_argument("--save_traj", type=str, help="Path to save trajectory in TUM format", default=None)

    args = parser.parse_args()

    view_reconstruction(
        args.filename,
        args.filter_threshold,
        args.filter_count,
        save_ply=args.save_ply,
        save_traj=args.save_traj
    )