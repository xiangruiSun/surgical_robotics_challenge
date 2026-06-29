#!/usr/bin/env python3
"""Collect synchronized cameraR RGB and PointCloud2 depth while keeping camera fixed.

This version keeps the static-camera collection behavior and revises the depth pipeline:
- It uses a strict AMBF -> OpenGL -> intrinsic triangulation depth conversion.
- It flips organized depth vertically after triangulation so the depth map matches ROS/OpenCV RGB image orientation.
- It does NOT move cameraR.
- It moves PSM1 and PSM2 with CRTK Cartesian servo_cp using POSITION ONLY.
- PSM orientation is always copied from the cached initial pose; no rotation command is changed.
- PSM translation follows a monotonic smooth ramp from zero offset to a tiny final x/y/z offset.
  This avoids the initial jump caused by sinusoidal cos() motion.
- Needle is not subscribed, commanded, held, or moved by this collector.
- It does NOT create depth_metric_mm/.

Outputs:
  image/frame_xxxxxx.png
  depth/frame_xxxxxx.npy
  depth/frame_xxxxxx.png
  depth_color/frame_xxxxxx.png
  metadata.csv

Run the slow 1000-frame collection:
  python3 collect_static_camera_position_only_needle_xy_1000.py

This script is tuned for slow PSM-only position motion: the PSM targets start exactly
at their initial poses and then slowly ramp by a tiny final x/y/z displacement across
all 1000 frames, with no commanded rotation. The needle is left untouched.
"""

import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from ambf_msgs.msg import CameraState
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def image_msg_to_cv2(bridge, msg):
    return np.asarray(bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"))



def read_xyz_points(msg):
    """Read x/y/z from PointCloud2 without dropping NaNs or changing point order."""
    field_names = {field.name for field in msg.fields}
    if not {"x", "y", "z"}.issubset(field_names):
        raise RuntimeError(f"PointCloud2 fields are {sorted(field_names)}, expected x/y/z")

    pts = point_cloud2.read_points_numpy(
        msg, field_names=["x", "y", "z"], skip_nans=False
    )
    pts = np.asarray(pts)
    if pts.dtype.names:
        pts = np.column_stack([pts["x"], pts["y"], pts["z"]])
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


def clean_metric_depth(depth, near_clip, far_clip):
    """Keep finite positive depth values in the requested metric range."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > float(near_clip))
    if far_clip > 0:
        valid &= depth < float(far_clip)
    out = np.zeros(depth.shape, dtype=np.float32)
    out[valid] = depth[valid]
    return out


# AMBF camera axes -> OpenCV camera axes, from the SRC projection example:
#   OpenCV: +X right, +Y down, +Z forward.
F_AMBF_TO_OPENCV = np.array([
    [0.0, 1.0,  0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0,  0.0, 1.0],
], dtype=np.float32)

# OpenCV camera axes -> OpenGL camera axes:
#   OpenCV: +X right, +Y down, +Z forward
#   OpenGL: +X right, +Y up,   -Z forward
# So x is unchanged, y and z are flipped.
F_OPENCV_TO_OPENGL = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float32)

# Final strict conversion used by this collector.
F_AMBF_TO_OPENGL = F_OPENCV_TO_OPENGL @ F_AMBF_TO_OPENCV
R_AMBF_TO_OPENGL = F_AMBF_TO_OPENGL[:3, :3]


def ambf_points_to_opengl(pts_ambf):
    """Convert Nx3 AMBF camera-frame points into OpenGL camera-frame points."""
    pts_ambf = np.asarray(pts_ambf, dtype=np.float32).reshape(-1, 3)
    return pts_ambf @ R_AMBF_TO_OPENGL.T


def triangulated_depth_from_opengl_points(pts_gl, height, width, fx, fy, cx, cy):
    """Compute per-pixel optical-axis depth from OpenGL camera points and K.

    OpenGL camera convention:
        +X right, +Y up, -Z forward.

    For image pixel (u, v), the OpenGL camera ray from the intrinsic matrix is:
        r = [(u - cx) / fx, -(v - cy) / fy, -1]

    A point on that ray satisfies:
        P_gl = depth * r

    Therefore depth is solved by least-squares triangulation:
        depth = dot(P_gl, r) / dot(r, r)

    This uses the camera intrinsics and the pixel location instead of choosing
    a depth convention by sign voting.
    """
    h, w = int(height), int(width)
    xyz = np.asarray(pts_gl, dtype=np.float32).reshape(h, w, 3)

    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]

    rx = (u - float(cx)) / float(fx)
    ry = -(v - float(cy)) / float(fy)
    rz = -1.0

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    denom = rx * rx + ry * ry + 1.0
    depth = (x * rx + y * ry + z * rz) / denom
    return depth.astype(np.float32, copy=False)


def project_unorganized_opengl_points_to_depth(pts_gl, height, width, fx, fy, cx, cy, near_clip, far_clip):
    """Fallback for an unorganized cloud using the same OpenGL + K model."""
    h, w = int(height), int(width)
    pts_gl = np.asarray(pts_gl, dtype=np.float32).reshape(-1, 3)
    finite = np.all(np.isfinite(pts_gl), axis=1)
    pts_gl = pts_gl[finite]
    if pts_gl.size == 0:
        return np.zeros((h, w), dtype=np.float32)

    x = pts_gl[:, 0]
    y = pts_gl[:, 1]
    z = pts_gl[:, 2]

    # OpenGL forward optical-axis depth.
    d = -z
    valid = d > float(near_clip)
    if far_clip > 0:
        valid &= d < float(far_clip)
    if not np.any(valid):
        return np.zeros((h, w), dtype=np.float32)

    x = x[valid]
    y = y[valid]
    d = d[valid]

    u = np.rint(float(cx) + float(fx) * x / d).astype(np.int32)
    v = np.rint(float(cy) - float(fy) * y / d).astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return np.zeros((h, w), dtype=np.float32)

    linear = v[inside].astype(np.int64) * w + u[inside].astype(np.int64)
    flat = np.full(h * w, np.inf, dtype=np.float32)
    np.minimum.at(flat, linear, d[inside].astype(np.float32, copy=False))
    depth = flat.reshape(h, w)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def pointcloud2_to_depth(
    msg,
    expected_height,
    expected_width,
    fx,
    fy,
    cx,
    cy,
    near_clip=0.001,
    far_clip=10.0,
    flip_vertical=True,
):
    """Strict AMBF -> OpenGL -> intrinsic-triangulated depth pipeline.

    Pipeline:
      1. Read AMBF PointCloud2 x/y/z.
      2. Convert AMBF camera coordinates to OpenGL camera coordinates using:
             F_AMBF_TO_OPENGL = F_OPENCV_TO_OPENGL @ F_AMBF_TO_OPENCV
      3. For each pixel, use K to build the OpenGL camera ray.
      4. Solve optical-axis depth by triangulation.
      5. Save a 2D metric depth map.
    """
    pts_ambf = read_xyz_points(msg)
    pts_gl = ambf_points_to_opengl(pts_ambf)

    h, w = int(expected_height), int(expected_width)
    n_expected = h * w

    if pts_gl.shape[0] == n_expected:
        raw_depth = triangulated_depth_from_opengl_points(
            pts_gl, h, w, fx, fy, cx, cy
        )
        # AMBF/OpenGL camera buffers are bottom-left origin while ROS/OpenCV
        # RGB images are top-left origin. Flip organized depth vertically so
        # depth[v, u] corresponds to the same pixel as RGB[v, u].
        if flip_vertical:
            raw_depth = np.flipud(raw_depth)
        method = "organized-ambf-to-opengl-intrinsic-triangulation-flipY" if flip_vertical else "organized-ambf-to-opengl-intrinsic-triangulation-noFlip"
    else:
        raw_depth = project_unorganized_opengl_points_to_depth(
            pts_gl, h, w, fx, fy, cx, cy, near_clip, far_clip
        )
        method = "projected-ambf-to-opengl-intrinsic-triangulation"

    depth = clean_metric_depth(raw_depth, near_clip, far_clip)
    valid_fraction = float(np.mean(depth > 0.0))

    xyz_min_ambf = np.nanmin(pts_ambf, axis=0) if pts_ambf.size else np.zeros(3)
    xyz_max_ambf = np.nanmax(pts_ambf, axis=0) if pts_ambf.size else np.zeros(3)
    xyz_min_gl = np.nanmin(pts_gl, axis=0) if pts_gl.size else np.zeros(3)
    xyz_max_gl = np.nanmax(pts_gl, axis=0) if pts_gl.size else np.zeros(3)

    if valid_fraction < 1e-4:
        raise RuntimeError(
            "Depth conversion produced an empty image. "
            f"method={method}, valid_fraction={valid_fraction:.6f}, "
            f"points={pts_ambf.shape[0]}, expected={n_expected}, "
            f"msg.height={msg.height}, msg.width={msg.width}, "
            f"point_step={msg.point_step}, row_step={msg.row_step}, "
            f"ambf_xyz_min={xyz_min_ambf.tolist()}, ambf_xyz_max={xyz_max_ambf.tolist()}, "
            f"opengl_xyz_min={xyz_min_gl.tolist()}, opengl_xyz_max={xyz_max_gl.tolist()}."
        )

    return np.ascontiguousarray(depth, dtype=np.float32), method


def depth_to_visible_gray(depth_m):
    """Viewer-only depth PNG. .npy remains metric source of truth."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    gray = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth_m[valid], [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
        # Same successful preview style: near bright, far dark, invalid black.
        gray[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return gray


def save_visible_depth_png(depth_m, gray_path, color_path):
    gray = depth_to_visible_gray(depth_m)
    if not cv2.imwrite(gray_path, gray):
        raise RuntimeError(f"Failed to write visible depth PNG: {gray_path}")

    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[gray == 0] = 0
    if not cv2.imwrite(color_path, color):
        raise RuntimeError(f"Failed to write colour depth PNG: {color_path}")


def copy_pose_stamped(src):
    dst = PoseStamped()
    dst.header = src.header
    dst.pose.position.x = src.pose.position.x
    dst.pose.position.y = src.pose.position.y
    dst.pose.position.z = src.pose.position.z
    dst.pose.orientation.x = src.pose.orientation.x
    dst.pose.orientation.y = src.pose.orientation.y
    dst.pose.orientation.z = src.pose.orientation.z
    dst.pose.orientation.w = src.pose.orientation.w
    return dst


class StaticCameraPSMNeedleCollector(Node):
    def __init__(self, args):
        super().__init__("static_camera_psm_needle_collector")
        self.args = args
        self.bridge = CvBridge()

        self.base_dir = os.path.expanduser(args.out_dir)
        self.image_dir = os.path.join(self.base_dir, "image")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.depth_color_dir = os.path.join(self.base_dir, "depth_color")
        for directory in (self.image_dir, self.depth_dir, self.depth_color_dir):
            os.makedirs(directory, exist_ok=True)

        self.rgb_queue = deque(maxlen=args.sync_queue_size)
        self.depth_queue = deque(maxlen=args.sync_queue_size)
        self.latest_camera_state = None
        self.latest_psm1_cp = None
        self.latest_psm2_cp = None

        self.initial_psm1_cp = None
        self.initial_psm2_cp = None

        self.count = 0
        self.last_saved_rgb_stamp = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(10, args.sync_queue_size),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.rgb_sub = self.create_subscription(Image, args.rgb_topic, self.rgb_cb, sensor_qos)
        self.depth_sub = self.create_subscription(PointCloud2, args.depth_topic, self.depth_cb, sensor_qos)
        self.camera_state_sub = self.create_subscription(
            CameraState, args.camera_state_topic, self.camera_state_cb, sensor_qos
        )

        self.psm1_cp_sub = self.create_subscription(
            PoseStamped, args.psm1_measured_cp_topic, self.psm1_cp_cb, sensor_qos
        )
        self.psm2_cp_sub = self.create_subscription(
            PoseStamped, args.psm2_measured_cp_topic, self.psm2_cp_cb, sensor_qos
        )
        self.psm1_servo_pub = self.create_publisher(PoseStamped, args.psm1_servo_cp_topic, reliable_qos)
        self.psm2_servo_pub = self.create_publisher(PoseStamped, args.psm2_servo_cp_topic, reliable_qos)

        self.metadata_path = os.path.join(self.base_dir, "metadata.csv")
        self.metadata_file = open(self.metadata_path, "w", newline="")
        self.metadata_writer = csv.writer(self.metadata_file)
        self.metadata_writer.writerow([
            "frame_id", "rgb_stamp_ns", "depth_stamp_ns", "sync_error_ms",
            "camera_stamp_ns", "rgb_file", "depth_npy_m", "depth_png_visible",
            "depth_color_png", "depth_source", "fx", "fy", "cx", "cy",
            "valid_fraction", "depth_method", "depth_min_m", "depth_median_m", "depth_max_m",
            "psm1_x", "psm1_y", "psm1_z",
            "psm2_x", "psm2_y", "psm2_z",
        ])

        self.get_logger().info(
            f"Collecting {args.num_frames} static-camera frames into {self.base_dir}"
        )
        self.get_logger().info(
            "Camera will NOT be commanded. PSM1/PSM2 move through CRTK servo_cp with a bounded XYZ waypoint path. "
            "PSM1/PSM2 move through CRTK servo_cp with POSITION ONLY: orientation is fixed. "
            "Both PSMs are published continuously during each frame wait so motion is visible. "
            "Needle is not commanded or moved by this collector."
        )
        self.get_logger().info(
            "Depth conversion is strict: AMBF camera points are transformed to OpenGL coordinates, K triangulates per-pixel optical-axis depth, then organized depth is vertically flipped to match OpenCV/RGB image orientation. No depth_metric_mm output."
        )

    def rgb_cb(self, msg):
        self.rgb_queue.append(msg)

    def depth_cb(self, msg):
        self.depth_queue.append(msg)

    def camera_state_cb(self, msg):
        self.latest_camera_state = msg

    def psm1_cp_cb(self, msg):
        self.latest_psm1_cp = msg

    def psm2_cp_cb(self, msg):
        self.latest_psm2_cp = msg

    def ready(self):
        return (
            bool(self.rgb_queue)
            and bool(self.depth_queue)
            and self.latest_camera_state is not None
            and self.latest_psm1_cp is not None
            and self.latest_psm2_cp is not None
        )

    def wait_until_ready(self):
        self.get_logger().info("Waiting for RGB, depth, camera state, and PSM1/PSM2 measured_cp...")
        while rclpy.ok() and not self.ready():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.initial_psm1_cp = copy_pose_stamped(self.latest_psm1_cp)
        self.initial_psm2_cp = copy_pose_stamped(self.latest_psm2_cp)

        self.get_logger().info("All required topics received. Cached initial poses.")

    def make_psm_target(self, initial_msg, dx, dy, dz):
        target = copy_pose_stamped(initial_msg)
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = float(initial_msg.pose.position.x + dx)
        target.pose.position.y = float(initial_msg.pose.position.y + dy)
        target.pose.position.z = float(initial_msg.pose.position.z + dz)
        # Orientation intentionally unchanged: position-only variation.
        return target


    def smoothstep(self, t):
        """Zero-velocity start/end ramp, t in [0, 1]."""
        t = max(0.0, min(1.0, float(t)))
        return t * t * (3.0 - 2.0 * t)

    def interpolate_waypoints(self, waypoints, t):
        """Smoothly interpolate a bounded waypoint path without any initial jump."""
        t = max(0.0, min(1.0, float(t)))
        nseg = len(waypoints) - 1
        if nseg <= 0:
            return np.zeros(3, dtype=float)
        scaled = t * nseg
        idx = min(int(scaled), nseg - 1)
        local_t = scaled - idx
        s = self.smoothstep(local_t)
        a = np.asarray(waypoints[idx], dtype=float)
        b = np.asarray(waypoints[idx + 1], dtype=float)
        return (1.0 - s) * a + s * b

    def publish_scene_targets(self):
        """Publish position-only targets for both PSMs along a bounded XYZ path.

        Movement strategy:
        - needle is not commanded at all;
        - PSM orientation is copied from the cached initial measured_cp;
        - PSM target starts exactly at initial pose;
        - each arm follows x, then y, then z, then returns toward start;
        - both PSM targets are republished during the frame wait for visible motion.
        """
        t = self.count / max(1.0, float(self.args.num_frames - 1))

        psm1_path = [
            (0.0, 0.0, 0.0),
            (self.args.psm1_dx_total, 0.0, 0.0),
            (self.args.psm1_dx_total, self.args.psm1_dy_total, 0.0),
            (self.args.psm1_dx_total, self.args.psm1_dy_total, self.args.psm1_dz_total),
            (0.0, 0.0, 0.0),
        ]
        psm2_path = [
            (0.0, 0.0, 0.0),
            (self.args.psm2_dx_total, 0.0, 0.0),
            (self.args.psm2_dx_total, self.args.psm2_dy_total, 0.0),
            (self.args.psm2_dx_total, self.args.psm2_dy_total, self.args.psm2_dz_total),
            (0.0, 0.0, 0.0),
        ]

        psm1_dx, psm1_dy, psm1_dz = self.interpolate_waypoints(psm1_path, t)
        psm2_dx, psm2_dy, psm2_dz = self.interpolate_waypoints(psm2_path, t)

        psm1_target = self.make_psm_target(self.initial_psm1_cp, psm1_dx, psm1_dy, psm1_dz)
        psm2_target = self.make_psm_target(self.initial_psm2_cp, psm2_dx, psm2_dy, psm2_dz)

        self.psm1_servo_pub.publish(psm1_target)
        self.psm2_servo_pub.publish(psm2_target)

        return psm1_target, psm2_target

    def publish_initial_scene(self):
        psm1_target = self.make_psm_target(self.initial_psm1_cp, 0.0, 0.0, 0.0)
        psm2_target = self.make_psm_target(self.initial_psm2_cp, 0.0, 0.0, 0.0)
        self.psm1_servo_pub.publish(psm1_target)
        self.psm2_servo_pub.publish(psm2_target)

    def closest_synchronized_pair(self):
        if not self.rgb_queue or not self.depth_queue:
            return None

        for rgb_msg in reversed(self.rgb_queue):
            rgb_ns = stamp_to_ns(rgb_msg.header.stamp)
            if rgb_ns == self.last_saved_rgb_stamp:
                continue
            depth_msg = min(
                self.depth_queue,
                key=lambda msg: abs(stamp_to_ns(msg.header.stamp) - rgb_ns),
            )
            depth_ns = stamp_to_ns(depth_msg.header.stamp)
            error_ms = abs(depth_ns - rgb_ns) / 1e6
            if error_ms <= self.args.max_sync_ms:
                return rgb_msg, depth_msg, error_ms
        return None

    def save_synchronized_pair(self, psm1_target, psm2_target):
        pair = self.closest_synchronized_pair()
        if pair is None:
            return False

        rgb_msg, depth_msg, sync_error_ms = pair
        rgb_stamp = stamp_to_ns(rgb_msg.header.stamp)
        frame_id = f"frame_{self.count:06d}"

        rgb = image_msg_to_cv2(self.bridge, rgb_msg)
        height, width = rgb.shape[:2]

        depth_m, depth_method = pointcloud2_to_depth(
            depth_msg,
            expected_height=height,
            expected_width=width,
            fx=self.args.fx,
            fy=self.args.fy,
            cx=self.args.cx,
            cy=self.args.cy,
            near_clip=self.args.near_clip,
            far_clip=self.args.far_clip,
            flip_vertical=self.args.flip_vertical,
        )

        rgb_file = os.path.join(self.image_dir, f"{frame_id}.png")
        depth_npy = os.path.join(self.depth_dir, f"{frame_id}.npy")
        depth_visible_png = os.path.join(self.depth_dir, f"{frame_id}.png")
        depth_color_png = os.path.join(self.depth_color_dir, f"{frame_id}.png")

        if not cv2.imwrite(rgb_file, rgb):
            raise RuntimeError(f"Failed to write RGB image: {rgb_file}")
        np.save(depth_npy, depth_m.astype(np.float32, copy=False))
        save_visible_depth_png(depth_m, depth_visible_png, depth_color_png)

        valid = depth_m > 0.0
        if np.any(valid):
            stats = (
                float(valid.mean()),
                float(depth_m[valid].min()),
                float(np.median(depth_m[valid])),
                float(depth_m[valid].max()),
            )
        else:
            stats = (0.0, 0.0, 0.0, 0.0)

        camera_msg = self.latest_camera_state
        camera_stamp = (
            stamp_to_ns(camera_msg.header.stamp)
            if camera_msg is not None and hasattr(camera_msg, "header") else ""
        )

        self.metadata_writer.writerow([
            frame_id,
            rgb_stamp,
            stamp_to_ns(depth_msg.header.stamp),
            f"{sync_error_ms:.6f}",
            camera_stamp,
            rgb_file,
            depth_npy,
            depth_visible_png,
            depth_color_png,
            "pointcloud",
            f"{self.args.fx:.12f}",
            f"{self.args.fy:.12f}",
            f"{self.args.cx:.12f}",
            f"{self.args.cy:.12f}",
            f"{stats[0]:.6f}",
            depth_method,
            f"{stats[1]:.6f}",
            f"{stats[2]:.6f}",
            f"{stats[3]:.6f}",
            f"{psm1_target.pose.position.x:.9f}",
            f"{psm1_target.pose.position.y:.9f}",
            f"{psm1_target.pose.position.z:.9f}",
            f"{psm2_target.pose.position.x:.9f}",
            f"{psm2_target.pose.position.y:.9f}",
            f"{psm2_target.pose.position.z:.9f}",
        ])
        self.metadata_file.flush()

        self.last_saved_rgb_stamp = rgb_stamp
        self.count += 1
        self.get_logger().info(
            f"Saved {frame_id} ({self.count}/{self.args.num_frames}), "
            f"sync={sync_error_ms:.2f} ms, valid={stats[0]:.1%}, "
            f"median_depth={stats[2]:.4f} m, method={depth_method}"
        )
        return True

    def collect(self):
        self.wait_until_ready()
        dt = 1.0 / self.args.rate_hz
        attempts_without_save = 0

        while rclpy.ok() and self.count < self.args.num_frames:
            psm1_target, psm2_target = self.publish_scene_targets()

            # Allow the simulator to update. Re-publish both PSM servo_cp targets
            # continuously during the wait; otherwise some CRTK cycles may not visibly move.
            deadline = time.monotonic() + self.args.frame_wait_sec
            saved = False
            next_hold = 0.0
            while rclpy.ok() and time.monotonic() < deadline and not saved:
                now = time.monotonic()
                if now >= next_hold:
                    self.psm1_servo_pub.publish(psm1_target)
                    self.psm2_servo_pub.publish(psm2_target)
                    next_hold = now + self.args.hold_publish_period
                rclpy.spin_once(self, timeout_sec=0.01)
                saved = self.save_synchronized_pair(psm1_target, psm2_target)

            if not saved:
                attempts_without_save += 1
                if attempts_without_save % 10 == 0:
                    self.get_logger().warning("No synchronized pair yet. Consider increasing --max_sync_ms.")
            else:
                attempts_without_save = 0

            if self.args.sleep_sec > 0:
                time.sleep(self.args.sleep_sec)

        self.get_logger().info("Returning PSM1/PSM2 to cached initial poses.")
        for _ in range(max(5, int(self.args.rate_hz))):
            self.publish_initial_scene()
            rclpy.spin_once(self, timeout_sec=dt)

        self.metadata_file.close()
        self.get_logger().info(f"Finished. Saved {self.count} frames to {self.base_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="training_data_static_camera_psm_xyz_waypoint_no_needle_1000")
    parser.add_argument("--num_frames", type=int, default=1000)

    parser.add_argument("--rgb_topic", default="/ambf/env/cameras/cameraR/ImageData")
    parser.add_argument("--depth_topic", default="/ambf/env/cameras/cameraR/DepthData")
    parser.add_argument("--camera_state_topic", default="/ambf/env/cameras/cameraR/State")
    parser.add_argument("--psm1_measured_cp_topic", default="/CRTK/psm1/measured_cp")
    parser.add_argument("--psm2_measured_cp_topic", default="/CRTK/psm2/measured_cp")
    parser.add_argument("--psm1_servo_cp_topic", default="/CRTK/psm1/servo_cp")
    parser.add_argument("--psm2_servo_cp_topic", default="/CRTK/psm2/servo_cp")

    parser.add_argument("--fx", type=float, default=358.8070272987445)
    parser.add_argument("--fy", type=float, default=358.8070272987445)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--near_clip", type=float, default=0.001)
    parser.add_argument("--far_clip", type=float, default=10.0)
    parser.add_argument(
        "--no_flip_vertical",
        dest="flip_vertical",
        action="store_false",
        help="Disable vertical flip after organized depth triangulation.",
    )
    parser.set_defaults(flip_vertical=True)
    # Position-only motion. These are TOTAL offsets reached only at the end of 1000 frames.
    # The target starts at zero offset and uses smoothstep, so the first frame does not jump.
    parser.add_argument("--psm1_dx_total", type=float, default=-0.01000)
    parser.add_argument("--psm1_dy_total", type=float, default=0.00600)
    parser.add_argument("--psm1_dz_total", type=float, default=0.00400)
    parser.add_argument("--psm2_dx_total", type=float, default=0.01000)
    parser.add_argument("--psm2_dy_total", type=float, default=-0.00600)
    parser.add_argument("--psm2_dz_total", type=float, default=0.00400)
    parser.add_argument("--rate_hz", type=float, default=20.0)
    parser.add_argument("--hold_publish_period", type=float, default=0.02)
    parser.add_argument("--frame_wait_sec", type=float, default=0.30)
    parser.add_argument("--sleep_sec", type=float, default=0.01)
    parser.add_argument("--max_sync_ms", type=float, default=20.0)
    parser.add_argument("--sync_queue_size", type=int, default=60)
    args = parser.parse_args()

    rclpy.init()
    node = StaticCameraPSMNeedleCollector(args)
    try:
        node.collect()
    except KeyboardInterrupt:
        node.get_logger().warning("Collection interrupted by user.")
    finally:
        if not node.metadata_file.closed:
            node.metadata_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
