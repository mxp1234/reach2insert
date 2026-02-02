#!/usr/bin/env python3
"""
Real-time Camera Calibration View

Displays the cropped camera feed for object position calibration.

Usage:
    python utils/camera_calibration_view.py
    python utils/camera_calibration_view.py --camera top
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serl_robot_infra", "franka_env"))

try:
    from franka_env.camera.rs_capture import RSCapture
    HAS_RS_CAPTURE = True
except ImportError:
    HAS_RS_CAPTURE = False
    print("Warning: RSCapture not available")

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("Warning: pyrealsense2 not available")


# Camera configurations from config.py
CAMERA_CONFIG = {
    "wrist_1": {
        "serial_number": "126122270333",
        "dim": (1280, 720),
        "exposure": 60000,
    },
    "wrist_2": {
        "serial_number": "315122270814",
        "dim": (1280, 720),
        "exposure": 50000,
    },
    "side": {
        "serial_number": "334622072595",
        "dim": (1280, 720),
        "exposure": 255,
    },
    "top": {
        "serial_number": "334622072595",
        "dim": (1280, 720),
        "exposure": 255,
    },
}

# Crop configurations from config.py
CROP_CONFIG = {
    "wrist_1": lambda img: img,
    "wrist_2": lambda img: img,
    "side": lambda img: img,
    "top": lambda img: img[145:261, 320:418],  # Object-centric crop
}


def list_available_cameras():
    """List all connected RealSense cameras."""
    if not HAS_REALSENSE:
        print("pyrealsense2 not available")
        return []

    ctx = rs.context()
    devices = ctx.devices
    print(f"\nFound {len(devices)} RealSense device(s):")
    serials = []
    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        serials.append(serial)
        print(f"  [{i}] {name} - Serial: {serial}")
    return serials


def run_calibration_view(camera_name: str = "top", show_full: bool = True):
    """
    Run real-time calibration view.

    Args:
        camera_name: Camera to use (top, wrist_1, wrist_2, side)
        show_full: Also show full frame alongside cropped view
    """
    if camera_name not in CAMERA_CONFIG:
        print(f"Unknown camera: {camera_name}")
        print(f"Available cameras: {list(CAMERA_CONFIG.keys())}")
        return

    cam_config = CAMERA_CONFIG[camera_name]
    crop_fn = CROP_CONFIG[camera_name]

    print(f"\n=== Camera Calibration View ===")
    print(f"Camera: {camera_name}")
    print(f"Serial: {cam_config['serial_number']}")
    print(f"Resolution: {cam_config['dim']}")

    # List available cameras
    available_serials = list_available_cameras()

    if cam_config['serial_number'] not in available_serials:
        print(f"\nError: Camera serial {cam_config['serial_number']} not found!")
        print("Please check camera connections.")
        return

    print(f"\nControls:")
    print("  Q/ESC : Quit")
    print("  S     : Save screenshot")
    print("  F     : Toggle full frame view")
    print("=" * 35)

    camera = None

    # Try using RSCapture from the project
    if HAS_RS_CAPTURE:
        try:
            camera = RSCapture(
                name=camera_name,
                serial_number=cam_config['serial_number'],
                dim=cam_config['dim'],
                fps=30,
                exposure=cam_config['exposure'],
            )
            print(f"Camera started successfully")
        except Exception as e:
            print(f"Failed to start camera with RSCapture: {e}")
            camera = None

    # Fallback to direct pyrealsense2
    if camera is None and HAS_REALSENSE:
        print("Trying direct pyrealsense2...")
        # Try different resolutions
        resolutions = [
            cam_config['dim'],  # Original
            (640, 480),         # Standard
            (848, 480),         # D435I native
            (1280, 720),        # HD
        ]
        for res in resolutions:
            try:
                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(cam_config['serial_number'])
                config.enable_stream(
                    rs.stream.color,
                    res[0],
                    res[1],
                    rs.format.bgr8,
                    30
                )
                pipeline.start(config)
                print(f"Camera started at {res[0]}x{res[1]}")
                break
            except Exception as e:
                print(f"Failed at {res[0]}x{res[1]}: {e}")
                pipeline = None
                continue

        if pipeline is None:
            print("Could not start camera at any resolution")
        else:
            class SimplePipeline:
                def __init__(self, pipe):
                    self.pipe = pipe

                def read(self):
                    frames = self.pipe.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        return True, np.asanyarray(color_frame.get_data())
                    return False, None

                def close(self):
                    self.pipe.stop()

            camera = SimplePipeline(pipeline)

            # Scale crop coordinates if resolution changed
            actual_res = res
            if actual_res != cam_config['dim']:
                print(f"Note: Using different resolution, crop region scaled accordingly")
                scale_x = actual_res[0] / cam_config['dim'][0]
                scale_y = actual_res[1] / cam_config['dim'][1]
                if camera_name == "top":
                    # Original crop: img[145:261, 320:418]
                    y1 = int(145 * scale_y)
                    y2 = int(261 * scale_y)
                    x1 = int(320 * scale_x)
                    x2 = int(418 * scale_x)
                    crop_fn = lambda img, y1=y1, y2=y2, x1=x1, x2=x2: img[y1:y2, x1:x2]
                    print(f"  Scaled crop region: img[{y1}:{y2}, {x1}:{x2}]")

    if camera is None:
        print("Error: Could not open camera")
        return

    cv2.namedWindow("Cropped View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cropped View", 400, 400)

    if show_full:
        cv2.namedWindow("Full Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Full Frame", 800, 600)

    screenshot_count = 0

    try:
        while True:
            # Read frame
            ret, frame = camera.read()
            if not ret or frame is None:
                continue

            # Apply crop
            cropped = crop_fn(frame)

            # Draw crosshair on cropped view (center reference)
            cropped_display = cropped.copy()
            h, w = cropped_display.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(cropped_display, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
            cv2.line(cropped_display, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

            # Add info text
            cv2.putText(cropped_display, f"Size: {w}x{h}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.imshow("Cropped View", cropped_display)

            # Show full frame with crop region highlighted
            if show_full:
                full_display = frame.copy()
                # Draw crop region rectangle (for top camera: [145:261, 320:418])
                if camera_name == "top":
                    cv2.rectangle(full_display, (320, 145), (418, 261), (0, 255, 0), 2)
                    cv2.putText(full_display, "Crop Region", (320, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Full Frame", full_display)

            # Handle key input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('s'):  # Save screenshot
                screenshot_count += 1
                filename = f"calibration_{camera_name}_{screenshot_count}.png"
                cv2.imwrite(filename, cropped)
                print(f"Saved: {filename}")
            elif key == ord('f'):  # Toggle full frame
                show_full = not show_full
                if show_full:
                    cv2.namedWindow("Full Frame", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Full Frame", 800, 600)
                else:
                    cv2.destroyWindow("Full Frame")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Camera calibration view")
    parser.add_argument('--camera', '-c', default='top',
                        choices=['top', 'wrist_1', 'wrist_2', 'side'],
                        help='Camera to use (default: top)')
    parser.add_argument('--no-full', action='store_true',
                        help='Hide full frame view')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available cameras and exit')
    args = parser.parse_args()

    if args.list:
        list_available_cameras()
        return

    run_calibration_view(camera_name=args.camera, show_full=not args.no_full)


if __name__ == "__main__":
    main()
