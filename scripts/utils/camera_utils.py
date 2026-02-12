"""
Camera system utilities.

Includes RealSense camera management and image processing.
"""

import cv2
import numpy as np
import threading
from typing import Dict, Optional

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available. Camera features disabled.")


# Default camera configuration
CAMERA_WARMUP_FRAMES = 30


class RealSenseCamera:
    """
    Single RealSense camera with threaded frame capture.
    """

    def __init__(self, serial: str, width: int, height: int, fps: int,
                 exposure: int = None, gain: int = None,
                 crop_width: list = None, crop_height: list = None):
        """
        Args:
            serial: Camera serial number
            width: Frame width
            height: Frame height
            fps: Frames per second
            exposure: Manual exposure in microseconds (None = auto)
            gain: Manual gain (None = auto)
            crop_width: [left_ratio, right_ratio] to crop from width (0-1)
            crop_height: [top_ratio, bottom_ratio] to crop from height (0-1)
        """
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.gain = gain
        self.crop_width = crop_width or [0, 0]
        self.crop_height = crop_height or [0, 0]
        self.pipeline = None
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.thread: Optional[threading.Thread] = None

    def start(self):
        """Start camera capture."""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not available")

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = self.pipeline.start(cfg)

        # Configure manual exposure if specified
        if self.exposure is not None or self.gain is not None:
            device = profile.get_device()
            for sensor in device.sensors:
                if sensor.get_info(rs.camera_info.name) == 'RGB Camera' or sensor.supports(rs.option.exposure):
                    if self.exposure is not None:
                        if sensor.supports(rs.option.enable_auto_exposure):
                            sensor.set_option(rs.option.enable_auto_exposure, 0)
                        if sensor.supports(rs.option.exposure):
                            sensor.set_option(rs.option.exposure, self.exposure)
                            print(f"    Exposure: {self.exposure}")
                    if self.gain is not None and sensor.supports(rs.option.gain):
                        sensor.set_option(rs.option.gain, self.gain)
                        print(f"    Gain: {self.gain}")
                    break

        # Warmup
        for _ in range(CAMERA_WARMUP_FRAMES):
            self.pipeline.wait_for_frames()

        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _apply_crop(self, img: np.ndarray) -> np.ndarray:
        """Apply ratio-based cropping to image."""
        h, w = img.shape[:2]
        left = int(w * self.crop_width[0])
        right = int(w * (1 - self.crop_width[1]))
        top = int(h * self.crop_height[0])
        bottom = int(h * (1 - self.crop_height[1]))
        return img[top:bottom, left:right]

    def _loop(self):
        """Background frame capture loop."""
        while not self._stop.is_set():
            frames = self.pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color:
                img = np.asanyarray(color.get_data())
                # Apply crop if configured
                if any(self.crop_width) or any(self.crop_height):
                    img = self._apply_crop(img)
                with self.lock:
                    self.latest_frame = img

    def read(self) -> Optional[np.ndarray]:
        """Get latest frame."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        """Stop camera capture."""
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.pipeline:
            self.pipeline.stop()


class MultiCameraSystem:
    """
    Multi-camera system manager.
    """

    def __init__(self, serials: Dict[str, str], width: int, height: int, fps: int,
                 exposure_config: Dict[str, Dict[str, int]] = None,
                 crop_config: Dict[str, Dict[str, list]] = None):
        """
        Args:
            serials: Dict mapping camera name to serial number
            width: Frame width
            height: Frame height
            fps: Frames per second
            exposure_config: Dict mapping camera name to {"exposure": int, "gain": int}
            crop_config: Dict mapping camera name to {"crop_width": [l, r], "crop_height": [t, b]}
        """
        exposure_config = exposure_config or {}
        crop_config = crop_config or {}
        self.cameras = {}
        for name, serial in serials.items():
            exp_cfg = exposure_config.get(name) or {}
            crop_cfg = crop_config.get(name) or {}
            exposure = exp_cfg.get("exposure") if exp_cfg else None
            gain = exp_cfg.get("gain") if exp_cfg else None
            crop_width = crop_cfg.get("crop_width") if crop_cfg else None
            crop_height = crop_cfg.get("crop_height") if crop_cfg else None
            self.cameras[name] = RealSenseCamera(
                serial, width, height, fps,
                exposure=exposure, gain=gain,
                crop_width=crop_width, crop_height=crop_height
            )
        self.failed_cameras = []

    def start(self):
        """Start all cameras."""
        self.failed_cameras = []
        for name, cam in self.cameras.items():
            try:
                cam.start()
                print(f"  Camera {name}: OK")
            except Exception as e:
                print(f"  Camera {name}: FAILED - {e}")
                self.failed_cameras.append(name)

    def all_cameras_ok(self) -> bool:
        """Check if all cameras started successfully."""
        return len(self.failed_cameras) == 0

    def read_all(self) -> Dict[str, Optional[np.ndarray]]:
        """Read frames from all cameras."""
        return {name: cam.read() for name, cam in self.cameras.items()}

    def stop(self):
        """Stop all cameras."""
        for cam in self.cameras.values():
            cam.stop()


def process_image_dp(
    img: Optional[np.ndarray],
    target_h: int,
    target_w: int,
    jpeg_quality: int = 90
) -> Optional[np.ndarray]:
    """
    Process image for DP inference.

    Args:
        img: Input BGR image
        target_h: Target height
        target_w: Target width
        jpeg_quality: JPEG compression quality

    Returns:
        Processed image (C, H, W) float32 normalized to [0, 1], RGB
    """
    if img is None:
        return None

    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # JPEG compression/decompression for consistency with training
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, img_encoded = cv2.imencode('.jpg', img, encode_param)
    img_decoded = cv2.imdecode(img_encoded, cv2.IMREAD_COLOR)

    # BGR to RGB, HWC to CHW, normalize
    img = cv2.cvtColor(img_decoded, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    return img


def process_image_serl(
    img: Optional[np.ndarray],
    h: int,
    w: int
) -> np.ndarray:
    """
    Process image for SERL.

    Args:
        img: Input BGR image
        h: Target height
        w: Target width

    Returns:
        Processed image (H, W, C) uint8, RGB
    """
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    img = cv2.resize(img, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)
