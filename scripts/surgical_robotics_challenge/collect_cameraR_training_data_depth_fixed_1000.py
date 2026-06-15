#!/usr/bin/env python3
"""Collect synchronized cameraR RGB and AMBF depth data.

AMBF publishes camera DepthData as sensor_msgs/PointCloud2. The cloud is in the
OpenGL optical convention: +X right, +Y up, and the camera looks along -Z.
Therefore metric optical-axis depth is -Z. OpenGL images/cloud rows also commonly
use a bottom-left origin, while ROS/OpenCV images use a top-left origin, so the
the XYZ samples are projected with the calibrated camera intrinsic matrix.

Outputs per frame:
  image/frame_xxxxxx.png                  BGR/RGB camera image
  depth/frame_xxxxxx.npy                  float32 metric depth in metres
  depth/frame_xxxxxx.png                  visible 8-bit grayscale depth image
  depth_color/frame_xxxxxx.png            visible colour-mapped depth image
  depth_metric_mm/frame_xxxxxx.png        uint16 metric depth in millimetres
  metadata.csv                            timestamps and conversion settings
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

from ambf_msgs.msg import CameraCmd, CameraState
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quat_from_euler(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    q = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float64)


def image_msg_to_cv2(bridge, msg):
    """Convert ROS image while respecting its declared encoding."""
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    return np.asarray(img)


def depth_image_to_numpy(bridge, msg):
    """Convert a ROS depth Image to metres following REP-118 conventions."""
    depth = np.asarray(
        bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    )
    if msg.encoding in ("16UC1", "mono16"):
        depth_m = depth.astype(np.float32) * 0.001
    elif msg.encoding == "32FC1":
        depth_m = depth.astype(np.float32)
    else:
        raise RuntimeError(
            f"Unsupported depth Image encoding {msg.encoding!r}; "
            "expected 16UC1 or 32FC1."
        )
    depth_m[~np.isfinite(depth_m)] = 0.0
    depth_m[depth_m <= 0.0] = 0.0
    return depth_m


def _read_xyz_points(msg):
    """Read x/y/z from PointCloud2 without dropping NaNs or pixel positions."""
    field_names = {field.name for field in msg.fields}
    if not {"x", "y", "z"}.issubset(field_names):
        raise RuntimeError(
            f"PointCloud2 fields are {sorted(field_names)}, expected x/y/z"
        )

    # read_points_numpy is the official ROS 2 path for equally typed XYZ fields.
    pts = point_cloud2.read_points_numpy(
        msg, field_names=["x", "y", "z"], skip_nans=False
    )
    pts = np.asarray(pts)
    if pts.dtype.names:
        pts = np.column_stack([pts["x"], pts["y"], pts["z"]])
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    return pts


def _clean_metric_depth(depth, near_clip, far_clip):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > float(near_clip))
    if far_clip > 0:
        valid &= depth < float(far_clip)
    out = np.zeros(depth.shape, dtype=np.float32)
    out[valid] = depth[valid]
    return out


def _best_organized_depth(pts, height, width, near_clip, far_clip, flip_vertical):
    """Use PointCloud2's image ordering directly when it has H*W points.

    An organized point cloud already has one XYZ sample per pixel. Reprojecting
    such a cloud with K is unnecessary and can erase the whole image when the
    producer's camera-axis convention differs. We test +Z, -Z, and Euclidean
    range, then choose the candidate with the largest valid population.
    """
    if pts.shape[0] != int(height) * int(width):
        return None, "not-organized", 0.0

    xyz = pts.reshape(int(height), int(width), 3)
    candidates = {
        "+z": xyz[..., 2],
        "-z": -xyz[..., 2],
        "abs(z)": np.abs(xyz[..., 2]),
        "range": np.linalg.norm(xyz, axis=2),
    }

    best = None
    for name, raw in candidates.items():
        depth = _clean_metric_depth(raw, near_clip, far_clip)
        fraction = float(np.mean(depth > 0))
        if best is None or fraction > best[0]:
            best = (fraction, name, depth)

    fraction, name, depth = best
    if flip_vertical:
        depth = np.flipud(depth)
    return np.ascontiguousarray(depth), f"organized:{name}", fraction


def _best_intrinsic_projection(pts, height, width, fx, fy, cx, cy,
                               near_clip, far_clip):
    """Fallback for unordered clouds; search axis permutations and signs.

    This handles OpenGL (-Z forward, +Y up), ROS optical (+Z forward, +Y down),
    and AMBF builds that publish coordinates in another camera-axis ordering.
    """
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.size == 0:
        return np.zeros((height, width), np.float32), "projection:no-finite-points", 0.0

    axis_names = ("x", "y", "z")
    best = None
    # forward axis, horizontal axis, vertical axis must be distinct.
    for f_axis in range(3):
        remaining = [i for i in range(3) if i != f_axis]
        for h_axis, v_axis in (remaining, remaining[::-1]):
            for f_sign in (1.0, -1.0):
                d = f_sign * pts[:, f_axis]
                base = d > float(near_clip)
                if far_clip > 0:
                    base &= d < float(far_clip)
                if not np.any(base):
                    continue
                dv = d[base]
                hp = pts[base, h_axis]
                vp = pts[base, v_axis]
                for h_sign in (1.0, -1.0):
                    for v_sign in (1.0, -1.0):
                        u = np.rint(float(cx) + h_sign * float(fx) * hp / dv).astype(np.int32)
                        v = np.rint(float(cy) + v_sign * float(fy) * vp / dv).astype(np.int32)
                        inside = ((u >= 0) & (u < int(width)) &
                                  (v >= 0) & (v < int(height)))
                        if not np.any(inside):
                            continue
                        linear = v[inside].astype(np.int64) * int(width) + u[inside].astype(np.int64)
                        # Score unique occupied pixels, not merely point count.
                        occupied = np.unique(linear).size
                        score = occupied / float(int(height) * int(width))
                        if best is None or score > best[0]:
                            best = (score, f_axis, h_axis, v_axis,
                                    f_sign, h_sign, v_sign,
                                    linear, dv[inside])

    if best is None:
        return np.zeros((height, width), np.float32), "projection:no-valid-convention", 0.0

    score, fa, ha, va, fs, hs, vs, linear, dv = best
    flat = np.full(int(height) * int(width), np.inf, dtype=np.float32)
    np.minimum.at(flat, linear, dv.astype(np.float32, copy=False))
    depth = flat.reshape(int(height), int(width))
    depth[~np.isfinite(depth)] = 0.0
    method = (
        f"projection:forward={fs:+g}{axis_names[fa]},"
        f"u={hs:+g}{axis_names[ha]},v={vs:+g}{axis_names[va]}"
    )
    return np.ascontiguousarray(depth), method, float(score)


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
    """Convert AMBF PointCloud2 to a metric depth image robustly.

    Primary path: preserve organized point-cloud pixel ordering.
    Fallback path: project an unordered cloud with K while searching camera-axis
    permutations/signs. The function raises instead of silently saving black
    images when no useful depth can be recovered.
    """
    pts = _read_xyz_points(msg)
    h, w = int(expected_height), int(expected_width)

    depth, method, fraction = _best_organized_depth(
        pts, h, w, near_clip, far_clip, flip_vertical
    )
    # Prefer direct organized conversion whenever it recovers at least 1% of
    # pixels. It preserves the exact pixel correspondence supplied by AMBF.
    if depth is None or fraction < 0.01:
        depth, method, fraction = _best_intrinsic_projection(
            pts, h, w, fx, fy, cx, cy, near_clip, far_clip
        )

    valid = depth > 0
    valid_fraction = float(valid.mean())
    xyz_min = np.nanmin(pts, axis=0) if pts.size else np.zeros(3)
    xyz_max = np.nanmax(pts, axis=0) if pts.size else np.zeros(3)

    if valid_fraction < 1e-4:
        raise RuntimeError(
            "Depth conversion produced an empty image. "
            f"method={method}, points={pts.shape[0]}, "
            f"msg.height={msg.height}, msg.width={msg.width}, "
            f"point_step={msg.point_step}, row_step={msg.row_step}, "
            f"xyz_min={xyz_min.tolist()}, xyz_max={xyz_max.tolist()}. "
            "No files were saved for this frame."
        )

    return np.ascontiguousarray(depth, dtype=np.float32), method

def save_metric_depth_png(depth_m, path):
    """Save lossless uint16 depth in millimetres; 0 means invalid."""
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    scaled = np.rint(depth_m[valid] * 1000.0)
    depth_mm[valid] = np.clip(scaled, 1, np.iinfo(np.uint16).max).astype(np.uint16)
    if not cv2.imwrite(path, depth_mm):
        raise RuntimeError(f"Failed to write depth PNG: {path}")


def depth_to_visible_gray(depth_m):
    """Convert metric depth to a visible uint8 image using robust per-frame scaling.

    This is for inspection only. The .npy file remains the authoritative metric depth.
    Near pixels are bright, far pixels are dark, and invalid pixels are black.
    """
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    gray = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth_m[valid], [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
        gray[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    return gray


def save_visible_depth_png(depth_m, gray_path, color_path):
    """Save depth images that ordinary image viewers display correctly."""
    gray = depth_to_visible_gray(depth_m)
    if not cv2.imwrite(gray_path, gray):
        raise RuntimeError(f"Failed to write visible depth PNG: {gray_path}")

    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[gray == 0] = 0
    if not cv2.imwrite(color_path, color):
        raise RuntimeError(f"Failed to write colour depth PNG: {color_path}")


class CameraRTrainingCollector(Node):
    def __init__(self, args):
        super().__init__("cameraR_training_collector")
        self.args = args
        self.bridge = CvBridge()

        self.base_dir = os.path.expanduser(args.out_dir)
        self.image_dir = os.path.join(self.base_dir, "image")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.depth_color_dir = os.path.join(self.base_dir, "depth_color")
        self.metric_png_dir = os.path.join(self.base_dir, "depth_metric_mm")
        for directory in (
            self.image_dir, self.depth_dir, self.depth_color_dir, self.metric_png_dir
        ):
            os.makedirs(directory, exist_ok=True)

        self.rgb_queue = deque(maxlen=args.sync_queue_size)
        self.depth_queue = deque(maxlen=args.sync_queue_size)
        self.latest_camera_state = None
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
        depth_cls = PointCloud2 if args.depth_type == "pointcloud" else Image
        self.depth_sub = self.create_subscription(depth_cls, args.depth_topic, self.depth_cb, sensor_qos)
        self.camera_state_sub = self.create_subscription(
            CameraState, args.camera_state_topic, self.camera_state_cb, sensor_qos
        )
        self.camera_cmd_pub = self.create_publisher(CameraCmd, args.camera_cmd_topic, reliable_qos)

        self.metadata_path = os.path.join(self.base_dir, "metadata.csv")
        self.metadata_file = open(self.metadata_path, "w", newline="")
        self.metadata_writer = csv.writer(self.metadata_file)
        self.metadata_writer.writerow([
            "frame_id", "rgb_stamp_ns", "depth_stamp_ns", "sync_error_ms",
            "camera_stamp_ns", "rgb_file", "depth_npy_m", "depth_png_visible",
            "depth_color_png", "depth_png_mm", "depth_source", "fx", "fy", "cx", "cy",
            "valid_fraction",
            "depth_method", "depth_min_m", "depth_median_m", "depth_max_m",
        ])

        self.get_logger().info(
            f"Collecting {args.num_frames} frames into {self.base_dir}; "
            f"max RGB-depth offset={args.max_sync_ms:.1f} ms"
        )
        self.get_logger().info(
            f"Depth conversion: {args.depth_type}; organized-cloud ordering first, "
            f"intrinsic projection fallback; K=[[{args.fx},0,{args.cx}],[0,{args.fy},{args.cy}],[0,0,1]]"
        )
        self.get_logger().info(
            "depth/*.png is viewer-friendly uint8; depth/*.npy is metric metres; "
            "depth_metric_mm/*.png is uint16 millimetres."
        )

    def rgb_cb(self, msg):
        self.rgb_queue.append(msg)

    def depth_cb(self, msg):
        self.depth_queue.append(msg)

    def camera_state_cb(self, msg):
        self.latest_camera_state = msg

    def ready(self):
        return bool(self.rgb_queue and self.depth_queue and self.latest_camera_state is not None)

    def wait_until_ready(self):
        self.get_logger().info("Waiting for RGB, depth, and cameraR state...")
        while rclpy.ok() and not self.ready():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("All required topics received.")

    def publish_camera_pose(self, x, y, z, qx, qy, qz, qw):
        cmd = CameraCmd()
        if hasattr(cmd, "enable_position_controller"):
            cmd.enable_position_controller = True
        cmd.pose.position.x, cmd.pose.position.y, cmd.pose.position.z = float(x), float(y), float(z)
        cmd.pose.orientation.x, cmd.pose.orientation.y = float(qx), float(qy)
        cmd.pose.orientation.z, cmd.pose.orientation.w = float(qz), float(qw)
        self.camera_cmd_pub.publish(cmd)

    def closest_synchronized_pair(self):
        """Return the newest unused RGB and the depth message nearest in time."""
        if not self.rgb_queue or not self.depth_queue:
            return None

        # Work newest-to-oldest so camera motion does not pair against stale frames.
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

    def save_synchronized_pair(self):
        pair = self.closest_synchronized_pair()
        if pair is None:
            return False
        rgb_msg, depth_msg, sync_error_ms = pair
        rgb_stamp = stamp_to_ns(rgb_msg.header.stamp)
        frame_id = f"frame_{self.count:06d}"

        rgb = image_msg_to_cv2(self.bridge, rgb_msg)
        height, width = rgb.shape[:2]

        if self.args.depth_type == "pointcloud":
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
        else:
            depth_method = "ros-depth-image"
            depth_m = depth_image_to_numpy(self.bridge, depth_msg)
            if depth_m.shape != (height, width):
                raise RuntimeError(
                    f"Depth image shape {depth_m.shape} does not match RGB {(height, width)}"
                )

        rgb_file = os.path.join(self.image_dir, f"{frame_id}.png")
        depth_npy = os.path.join(self.depth_dir, f"{frame_id}.npy")
        depth_visible_png = os.path.join(self.depth_dir, f"{frame_id}.png")
        depth_color_png = os.path.join(self.depth_color_dir, f"{frame_id}.png")
        depth_metric_png = os.path.join(self.metric_png_dir, f"{frame_id}.png")

        if not cv2.imwrite(rgb_file, rgb):
            raise RuntimeError(f"Failed to write RGB image: {rgb_file}")
        np.save(depth_npy, depth_m.astype(np.float32, copy=False))
        save_visible_depth_png(depth_m, depth_visible_png, depth_color_png)
        save_metric_depth_png(depth_m, depth_metric_png)

        valid = depth_m > 0.0
        if np.any(valid):
            stats = (
                float(valid.mean()), float(depth_m[valid].min()),
                float(np.median(depth_m[valid])), float(depth_m[valid].max()),
            )
        else:
            stats = (0.0, 0.0, 0.0, 0.0)

        camera_msg = self.latest_camera_state
        camera_stamp = (
            stamp_to_ns(camera_msg.header.stamp)
            if camera_msg is not None and hasattr(camera_msg, "header") else ""
        )
        self.metadata_writer.writerow([
            frame_id, rgb_stamp, stamp_to_ns(depth_msg.header.stamp),
            f"{sync_error_ms:.6f}", camera_stamp, rgb_file, depth_npy,
            depth_visible_png, depth_color_png, depth_metric_png,
            self.args.depth_type,
            f"{self.args.fx:.12f}", f"{self.args.fy:.12f}",
            f"{self.args.cx:.12f}", f"{self.args.cy:.12f}",
            f"{stats[0]:.6f}", depth_method, f"{stats[1]:.6f}", f"{stats[2]:.6f}", f"{stats[3]:.6f}",
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

    def move_and_collect(self):
        self.wait_until_ready()
        start_pose = self.latest_camera_state.pose
        x0, y0, z0 = start_pose.position.x, start_pose.position.y, start_pose.position.z
        base_q = np.array([
            start_pose.orientation.x, start_pose.orientation.y,
            start_pose.orientation.z, start_pose.orientation.w,
        ], dtype=np.float64)
        base_q /= np.linalg.norm(base_q) + 1e-12

        dt = 1.0 / self.args.rate_hz
        attempts_without_save = 0

        while rclpy.ok() and self.count < self.args.num_frames:
            t = self.count / max(1.0, float(self.args.num_frames - 1))
            x = x0 + self.args.move_x * t
            y = y0 + self.args.move_y * math.sin(2.0 * math.pi * t)
            z = z0 + self.args.move_z * math.sin(math.pi * t)

            yaw = math.radians(self.args.yaw_deg) * math.sin(2.0 * math.pi * t)
            pitch = math.radians(self.args.pitch_deg) * math.cos(2.0 * math.pi * t)
            roll = math.radians(self.args.roll_deg) * math.sin(2.0 * math.pi * t)
            q = quat_multiply(base_q, quat_from_euler(roll, pitch, yaw))
            q /= np.linalg.norm(q) + 1e-12

            for _ in range(self.args.command_repeats):
                self.publish_camera_pose(x, y, z, *q)
                rclpy.spin_once(self, timeout_sec=dt)

            # Allow both streams to deliver samples corresponding to this pose.
            deadline = time.monotonic() + self.args.frame_wait_sec
            saved = False
            while rclpy.ok() and time.monotonic() < deadline and not saved:
                rclpy.spin_once(self, timeout_sec=0.01)
                saved = self.save_synchronized_pair()

            if not saved:
                attempts_without_save += 1
                if attempts_without_save % 10 == 0:
                    self.get_logger().warning(
                        "No synchronized pair yet. Consider increasing --max_sync_ms."
                    )
            else:
                attempts_without_save = 0

            if self.args.sleep_sec > 0:
                time.sleep(self.args.sleep_sec)

        self.get_logger().info("Returning cameraR to its original pose.")
        for _ in range(max(1, int(self.args.rate_hz))):
            self.publish_camera_pose(x0, y0, z0, *base_q)
            rclpy.spin_once(self, timeout_sec=dt)

        self.metadata_file.close()
        self.get_logger().info(f"Finished. Saved {self.count} frames to {self.base_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="training_data_cameraR_depth_fixed_1000")
    parser.add_argument("--num_frames", type=int, default=1000)

    parser.add_argument("--rgb_topic", default="/ambf/env/cameras/cameraR/ImageData")
    parser.add_argument("--depth_topic", default="/ambf/env/cameras/cameraR/DepthData")
    parser.add_argument("--camera_state_topic", default="/ambf/env/cameras/cameraR/State")
    parser.add_argument("--camera_cmd_topic", default="/ambf/env/cameras/cameraR/Command")
    parser.add_argument("--depth_type", choices=["image", "pointcloud"], default="pointcloud")

    parser.add_argument("--fx", type=float, default=358.8070272987445)
    parser.add_argument("--fy", type=float, default=358.8070272987445)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--near_clip", type=float, default=0.001)
    parser.add_argument("--far_clip", type=float, default=10.0)
    parser.add_argument(
        "--no_flip_vertical", dest="flip_vertical", action="store_false",
        help="Disable the default OpenGL-bottom-left to OpenCV-top-left row flip."
    )
    parser.set_defaults(flip_vertical=True)
    parser.add_argument("--max_sync_ms", type=float, default=20.0)
    parser.add_argument("--sync_queue_size", type=int, default=60)
    parser.add_argument("--frame_wait_sec", type=float, default=0.25)

    parser.add_argument("--move_x", type=float, default=-0.02)
    parser.add_argument("--move_y", type=float, default=0.01)
    parser.add_argument("--move_z", type=float, default=0.005)
    parser.add_argument("--yaw_deg", type=float, default=5.0)
    parser.add_argument("--pitch_deg", type=float, default=3.0)
    parser.add_argument("--roll_deg", type=float, default=0.0)
    parser.add_argument("--rate_hz", type=float, default=30.0)
    parser.add_argument("--command_repeats", type=int, default=3)
    parser.add_argument("--sleep_sec", type=float, default=0.02)
    args = parser.parse_args()

    rclpy.init()
    node = CameraRTrainingCollector(args)
    try:
        node.move_and_collect()
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
