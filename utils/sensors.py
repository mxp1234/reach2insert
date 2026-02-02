"""
传感器和相机工具类
"""

import time
import threading
import numpy as np
import cv2
import requests
from typing import Dict, Optional, Tuple
from collections import OrderedDict

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available")


class RealSenseCamera:
    """单个 RealSense 相机"""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = None
        self.thread = None

    def start(self, warmup_frames: int = 30):
        """启动相机"""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not available")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.pipeline.start(config)

        # 预热
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        """后台读取循环"""
        while not self._stop.is_set():
            try:
                frames = self.pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if color:
                    with self.lock:
                        self.latest_frame = np.asanyarray(color.get_data())
                        self.latest_timestamp = time.time()
            except:
                break

    def read(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """读取最新帧"""
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.latest_timestamp
            return None, None

    def stop(self):
        """停止相机"""
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pipeline:
            self.pipeline.stop()


class MultiCameraSystem:
    """多相机系统"""

    def __init__(self, config):
        """
        初始化多相机系统

        Args:
            config: CameraConfig 配置对象
        """
        self.config = config
        self.cameras: Dict[str, RealSenseCamera] = {}

    def start(self) -> Dict[str, bool]:
        """
        启动所有相机

        Returns:
            各相机启动状态
        """
        status = {}
        for name, serial in self.config.serials.items():
            try:
                cam = RealSenseCamera(
                    serial, self.config.width, self.config.height, self.config.fps
                )
                cam.start(self.config.warmup_frames)
                self.cameras[name] = cam
                status[name] = True
                print(f"  Camera {name}: OK")
            except Exception as e:
                status[name] = False
                print(f"  Camera {name}: FAILED - {e}")
        return status

    def read_all(self) -> Tuple[Dict[str, np.ndarray], float]:
        """
        读取所有相机的最新帧

        Returns:
            图像字典, 最早时间戳
        """
        result = {}
        timestamps = []
        for name, cam in self.cameras.items():
            frame, ts = cam.read()
            result[name] = frame
            if ts is not None:
                timestamps.append(ts)
        return result, min(timestamps) if timestamps else time.time()

    def stop(self):
        """停止所有相机"""
        for cam in self.cameras.values():
            cam.stop()


class RobotInterface:
    """机器人通信接口"""

    def __init__(self, server_url: str):
        self.url = server_url

    def get_state(self) -> Optional[Dict]:
        """获取机器人状态"""
        try:
            response = requests.post(f"{self.url}/getstate", timeout=0.5)
            if response.status_code == 200:
                state = response.json()
                return {
                    "tcp_pose": np.array(state["ee"], dtype=np.float32),
                    "gripper_pos": float(state.get("gripper_pos", 0.0)),
                    "tcp_force": np.array(state.get("force", [0, 0, 0]), dtype=np.float32),
                    "tcp_torque": np.array(state.get("torque", [0, 0, 0]), dtype=np.float32),
                    "tcp_vel": np.zeros(6, dtype=np.float32),  # 需要从服务器获取
                }
        except Exception as e:
            print(f"Warning: Failed to get robot state: {e}")
        return None

    def send_pose(self, pose: np.ndarray):
        """发送目标位姿"""
        try:
            requests.post(f"{self.url}/pose", json={"arr": pose.tolist()}, timeout=0.5)
        except:
            pass

    def open_gripper(self):
        """打开夹爪"""
        try:
            requests.post(f"{self.url}/open_gripper", timeout=0.5)
        except:
            pass

    def close_gripper(self):
        """闭合夹爪"""
        try:
            requests.post(f"{self.url}/close_gripper", timeout=0.5)
        except:
            pass

    def clear_error(self):
        """清除错误"""
        try:
            requests.post(f"{self.url}/clearerr", timeout=1.0)
        except:
            pass

    def update_param(self, param: Dict):
        """更新控制参数"""
        try:
            requests.post(f"{self.url}/update_param", json=param, timeout=0.5)
        except:
            pass


def process_image(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    jpeg_quality: int = 90
) -> np.ndarray:
    """
    处理图像: resize + JPEG 压缩 + 归一化

    Args:
        img: 输入图像 (H, W, 3) BGR
        target_h: 目标高度
        target_w: 目标宽度
        jpeg_quality: JPEG 质量

    Returns:
        处理后的图像 (3, H, W) RGB, float32, [0, 1]
    """
    if img is None:
        return None

    # Resize
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # JPEG 压缩 (模拟传输)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # (H, W, 3) -> (3, H, W), normalize
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    return img
