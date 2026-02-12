#!/usr/bin/env python3
"""
Camera Crop Configuration Tool.

Features:
1. Interactive crop selection from live camera
2. Save crop settings with reference images for calibration
3. Load and display previous crop settings for comparison
4. Compare current camera view with saved reference

Usage:
    # Select and save crop for a camera
    python -m scripts.utils.camera_crop --camera side --save

    # Load saved settings and adjust (shows reference for comparison)
    python -m scripts.utils.camera_crop --camera side --load

    # Compare current camera with saved reference
    python -m scripts.utils.camera_crop --camera side --compare

    # List all saved crop configurations
    python -m scripts.utils.camera_crop --list

    # Delete saved configuration
    python -m scripts.utils.camera_crop --camera side --delete

Controls:
    Mouse drag  : Select crop region
    C           : Input crop manually (e.g., "214:400,636:799")
    R           : Reset selection
    S           : Save current selection
    Enter/Space : Confirm selection
    Q/Escape    : Cancel
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Path setup
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)

# Default config directory
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_DIR, "configs", "camera_crops")

# Camera configurations
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
    "top": {
        "serial_number": "334622072595",
        "dim": (640, 480),
        "exposure": 255,
    },
    "side": {
        "serial_number": "334622072595",
        "dim": (1280, 720),
        "exposure": 255,
    },
}

# Global variables for mouse callback
drawing = False
start_point = None
end_point = None
current_rect = None
selected_rect = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for rectangle selection."""
    global drawing, start_point, end_point, current_rect, selected_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            current_rect = (start_point, end_point)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        if start_point and end_point:
            x1 = min(start_point[0], end_point[0])
            y1 = min(start_point[1], end_point[1])
            x2 = max(start_point[0], end_point[0])
            y2 = max(start_point[1], end_point[1])
            selected_rect = ((x1, y1), (x2, y2))
            current_rect = selected_rect


def parse_crop_string(crop_str: str) -> Tuple[int, int, int, int]:
    """
    Parse crop string like "214:400,636:799" or "[214:400, 636:799]"
    Returns (y1, y2, x1, x2)
    """
    crop_str = crop_str.strip().strip('[]').replace(' ', '')
    parts = crop_str.split(',')
    if len(parts) != 2:
        raise ValueError(f"Invalid crop format: {crop_str}. Expected 'y1:y2,x1:x2'")

    y_part, x_part = parts
    y1, y2 = map(int, y_part.split(':'))
    x1, x2 = map(int, x_part.split(':'))

    return (y1, y2, x1, x2)


def crop_to_string(crop_region: Tuple[int, int, int, int]) -> str:
    """Convert crop region tuple to string format."""
    y1, y2, x1, x2 = crop_region
    return f"{y1}:{y2},{x1}:{x2}"


class CropConfigManager:
    """Manages crop configuration saving and loading."""

    def __init__(self, config_dir: str = DEFAULT_CONFIG_DIR):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

    def _get_config_path(self, camera_name: str) -> str:
        return os.path.join(self.config_dir, f"{camera_name}_crop.json")

    def _get_ref_image_path(self, camera_name: str) -> str:
        return os.path.join(self.config_dir, f"{camera_name}_reference.png")

    def _get_cropped_ref_path(self, camera_name: str) -> str:
        return os.path.join(self.config_dir, f"{camera_name}_cropped_reference.png")

    def save_config(
        self,
        camera_name: str,
        crop_region: Tuple[int, int, int, int],
        reference_image: np.ndarray,
        image_shape: Tuple[int, int],
        notes: str = ""
    ) -> str:
        """
        Save crop configuration with reference image.

        Args:
            camera_name: Name of the camera
            crop_region: (y1, y2, x1, x2) crop region
            reference_image: Full frame reference image
            image_shape: (height, width) of the image
            notes: Optional notes about this configuration

        Returns:
            Path to saved config file
        """
        y1, y2, x1, x2 = crop_region

        config = {
            "camera_name": camera_name,
            "crop_region": {
                "y1": y1,
                "y2": y2,
                "x1": x1,
                "x2": x2,
                "crop_string": crop_to_string(crop_region),
            },
            "image_shape": {
                "height": image_shape[0],
                "width": image_shape[1],
            },
            "crop_size": {
                "height": y2 - y1,
                "width": x2 - x1,
            },
            "timestamp": datetime.now().isoformat(),
            "notes": notes,
            "reference_image": os.path.basename(self._get_ref_image_path(camera_name)),
            "cropped_reference": os.path.basename(self._get_cropped_ref_path(camera_name)),
        }

        # Save config JSON
        config_path = self._get_config_path(camera_name)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save reference image (full frame with crop overlay)
        ref_with_overlay = reference_image.copy()
        cv2.rectangle(ref_with_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(ref_with_overlay, f"Crop: [{y1}:{y2}, {x1}:{x2}]",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(self._get_ref_image_path(camera_name), ref_with_overlay)

        # Save cropped reference
        cropped = reference_image[y1:y2, x1:x2]
        cv2.imwrite(self._get_cropped_ref_path(camera_name), cropped)

        print(f"\n[Saved] Configuration saved to: {config_path}")
        print(f"[Saved] Reference image: {self._get_ref_image_path(camera_name)}")
        print(f"[Saved] Cropped reference: {self._get_cropped_ref_path(camera_name)}")

        return config_path

    def load_config(self, camera_name: str) -> Optional[Dict[str, Any]]:
        """
        Load crop configuration for a camera.

        Returns:
            Config dict or None if not found
        """
        config_path = self._get_config_path(camera_name)
        if not os.path.exists(config_path):
            return None

        with open(config_path, 'r') as f:
            config = json.load(f)

        return config

    def load_reference_image(self, camera_name: str) -> Optional[np.ndarray]:
        """Load the reference image for a camera."""
        ref_path = self._get_ref_image_path(camera_name)
        if os.path.exists(ref_path):
            return cv2.imread(ref_path)
        return None

    def load_cropped_reference(self, camera_name: str) -> Optional[np.ndarray]:
        """Load the cropped reference image for a camera."""
        ref_path = self._get_cropped_ref_path(camera_name)
        if os.path.exists(ref_path):
            return cv2.imread(ref_path)
        return None

    def get_crop_region(self, camera_name: str) -> Optional[Tuple[int, int, int, int]]:
        """Get saved crop region as tuple (y1, y2, x1, x2)."""
        config = self.load_config(camera_name)
        if config is None:
            return None

        cr = config["crop_region"]
        return (cr["y1"], cr["y2"], cr["x1"], cr["x2"])

    def delete_config(self, camera_name: str) -> bool:
        """Delete saved configuration for a camera."""
        deleted = False
        for path in [
            self._get_config_path(camera_name),
            self._get_ref_image_path(camera_name),
            self._get_cropped_ref_path(camera_name),
        ]:
            if os.path.exists(path):
                os.remove(path)
                print(f"[Deleted] {path}")
                deleted = True
        return deleted

    def list_configs(self) -> Dict[str, Dict]:
        """List all saved crop configurations."""
        configs = {}
        for camera_name in CAMERA_CONFIG.keys():
            config = self.load_config(camera_name)
            if config:
                configs[camera_name] = config
        return configs


def start_camera(camera_name: str):
    """Start a RealSense camera."""
    cfg = CAMERA_CONFIG.get(camera_name)
    if not cfg:
        raise ValueError(f"Unknown camera: {camera_name}")

    try:
        sys.path.insert(0, os.path.join(PROJECT_DIR, "serl_robot_infra", "franka_env"))
        from franka_env.camera.rs_capture import RSCapture

        camera = RSCapture(
            name=camera_name,
            serial_number=cfg["serial_number"],
            dim=cfg["dim"],
            fps=15,
            exposure=cfg["exposure"],
        )
        return camera
    except Exception as e:
        print(f"Error starting camera '{camera_name}': {e}")
        return None


def select_crop_with_reference(
    camera,
    camera_name: str,
    config_manager: CropConfigManager,
    initial_crop: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Interactive crop selection with reference image comparison.

    Shows:
    - Left: Saved reference image (if exists)
    - Right: Current live camera view

    Returns:
        Selected crop region (y1, y2, x1, x2) or None if cancelled
    """
    global drawing, start_point, end_point, current_rect, selected_rect

    # Reset global state
    drawing = False
    start_point = None
    end_point = None
    current_rect = None
    selected_rect = None

    # Load saved reference
    saved_config = config_manager.load_config(camera_name)
    saved_ref = config_manager.load_reference_image(camera_name)
    saved_crop_ref = config_manager.load_cropped_reference(camera_name)

    # Initialize crop from saved or provided
    if initial_crop:
        y1, y2, x1, x2 = initial_crop
        selected_rect = ((x1, y1), (x2, y2))
        current_rect = selected_rect
    elif saved_config:
        cr = saved_config["crop_region"]
        selected_rect = ((cr["x1"], cr["y1"]), (cr["x2"], cr["y2"]))
        current_rect = selected_rect

    # Get initial frame
    ret, frame = camera.read()
    if not ret:
        print("Failed to read from camera")
        return None

    h, w = frame.shape[:2]

    # Print info
    print(f"\n{'='*60}")
    print(f"  Crop Selector for '{camera_name}'")
    print(f"{'='*60}")
    print(f"Image size: {w}x{h}")

    if saved_config:
        cr = saved_config["crop_region"]
        print(f"\n[Saved Config] Last updated: {saved_config.get('timestamp', 'unknown')}")
        print(f"  Crop: img[{cr['y1']}:{cr['y2']}, {cr['x1']}:{cr['x2']}]")
        print(f"  Size: {saved_config['crop_size']['width']}x{saved_config['crop_size']['height']}")
        if saved_config.get('notes'):
            print(f"  Notes: {saved_config['notes']}")
    else:
        print("\n[No Saved Config] Starting fresh")

    print(f"\nControls:")
    print(f"  Mouse drag  : Select crop region")
    print(f"  C           : Input crop manually")
    print(f"  R           : Reset selection")
    print(f"  S           : Save current selection")
    print(f"  Enter/Space : Confirm and save")
    print(f"  Q/Escape    : Cancel")

    # Window setup
    if saved_ref is not None:
        window_name = "Crop Selector (Left: Saved Reference | Right: Live Camera)"
    else:
        window_name = "Crop Selector (Live Camera)"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    last_frame = None

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        last_frame = frame.copy()
        display = frame.copy()

        # Draw current/selected rectangle on live view
        if current_rect or selected_rect:
            rect = current_rect if current_rect else selected_rect
            p1, p2 = rect
            cv2.rectangle(display, p1, p2, (0, 255, 0), 2)

            # Show dimensions
            w_rect = abs(p2[0] - p1[0])
            h_rect = abs(p2[1] - p1[1])
            text = f"{w_rect}x{h_rect}"
            cv2.putText(display, text, (p1[0], p1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add info text
        cv2.putText(display, "LIVE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if selected_rect:
            p1, p2 = selected_rect
            crop_text = f"Crop: img[{p1[1]}:{p2[1]}, {p1[0]}:{p2[0]}]"
            cv2.putText(display, crop_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Create combined display with reference
        if saved_ref is not None:
            # Resize reference to match live frame if needed
            ref_display = saved_ref.copy()
            if ref_display.shape[:2] != display.shape[:2]:
                ref_display = cv2.resize(ref_display, (display.shape[1], display.shape[0]))

            cv2.putText(ref_display, "REFERENCE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            combined = np.hstack([ref_display, display])
        else:
            combined = display

        cv2.imshow(window_name, combined)

        # Show cropped preview windows
        if selected_rect:
            p1, p2 = selected_rect
            y1, y2, x1, x2 = p1[1], p2[1], p1[0], p2[0]

            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                live_cropped = frame[y1:y2, x1:x2]
                if live_cropped.size > 0:
                    cv2.imshow("Live Crop", live_cropped)

                    # Resize to 128x128 for preview
                    resized = cv2.resize(live_cropped, (128, 128))
                    cv2.imshow("128x128 Preview", resized)

                    # Show saved crop reference side by side if available
                    if saved_crop_ref is not None:
                        # Resize saved crop to match current crop size
                        saved_resized = cv2.resize(saved_crop_ref, (live_cropped.shape[1], live_cropped.shape[0]))
                        crop_compare = np.hstack([saved_resized, live_cropped])
                        cv2.putText(crop_compare, "Saved", (5, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        cv2.putText(crop_compare, "Live", (saved_resized.shape[1] + 5, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.imshow("Crop Comparison (Saved | Live)", crop_compare)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('r'):  # Reset
            current_rect = None
            selected_rect = None
            for win in ["Live Crop", "128x128 Preview", "Crop Comparison (Saved | Live)"]:
                try:
                    cv2.destroyWindow(win)
                except:
                    pass
            print("Selection reset")

        elif key == ord('c'):  # Manual input
            cv2.destroyAllWindows()
            crop_input = input("\nEnter crop range (e.g., '214:400,636:799'): ").strip()
            if crop_input:
                try:
                    y1, y2, x1, x2 = parse_crop_string(crop_input)
                    selected_rect = ((x1, y1), (x2, y2))
                    current_rect = selected_rect
                    print(f"Set crop: img[{y1}:{y2}, {x1}:{x2}]")
                except Exception as e:
                    print(f"Invalid crop format: {e}")
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, mouse_callback)

        elif key == ord('s'):  # Save
            if selected_rect and last_frame is not None:
                p1, p2 = selected_rect
                crop_region = (p1[1], p2[1], p1[0], p2[0])

                notes = input("\nEnter notes for this config (optional): ").strip()
                config_manager.save_config(
                    camera_name=camera_name,
                    crop_region=crop_region,
                    reference_image=last_frame,
                    image_shape=(h, w),
                    notes=notes,
                )

                # Reload reference
                saved_ref = config_manager.load_reference_image(camera_name)
                saved_crop_ref = config_manager.load_cropped_reference(camera_name)
            else:
                print("No crop region selected!")

        elif key == 13 or key == 32:  # Enter or Space - confirm and save
            if selected_rect:
                p1, p2 = selected_rect
                crop_region = (p1[1], p2[1], p1[0], p2[0])

                if last_frame is not None:
                    notes = input("\nEnter notes for this config (optional, press Enter to skip): ").strip()
                    config_manager.save_config(
                        camera_name=camera_name,
                        crop_region=crop_region,
                        reference_image=last_frame,
                        image_shape=(h, w),
                        notes=notes,
                    )

                cv2.destroyAllWindows()
                return crop_region
            else:
                print("No region selected!")

        elif key == ord('q') or key == 27:  # Q or Escape
            cv2.destroyAllWindows()
            return None

    return None


def compare_with_reference(
    camera,
    camera_name: str,
    config_manager: CropConfigManager,
):
    """
    Compare current camera view with saved reference side by side.
    Useful for checking if camera position has changed.
    """
    saved_config = config_manager.load_config(camera_name)
    saved_ref = config_manager.load_reference_image(camera_name)

    if saved_config is None or saved_ref is None:
        print(f"No saved configuration found for camera '{camera_name}'")
        return

    crop_region = config_manager.get_crop_region(camera_name)

    print(f"\n{'='*60}")
    print(f"  Camera Comparison: '{camera_name}'")
    print(f"{'='*60}")
    print(f"Saved config timestamp: {saved_config.get('timestamp', 'unknown')}")
    if saved_config.get('notes'):
        print(f"Notes: {saved_config['notes']}")
    print(f"\nControls:")
    print(f"  Q: Quit")

    window_name = "Comparison (Left: Saved Reference | Right: Live Camera)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        display = frame.copy()

        # Draw crop region on live
        if crop_region:
            y1, y2, x1, x2 = crop_region
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Resize reference to match
        ref_display = saved_ref.copy()
        if ref_display.shape[:2] != display.shape[:2]:
            ref_display = cv2.resize(ref_display, (display.shape[1], display.shape[0]))

        # Add labels
        cv2.putText(ref_display, "SAVED REFERENCE", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display, "LIVE CAMERA", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        combined = np.hstack([ref_display, display])
        cv2.imshow(window_name, combined)

        # Show cropped comparison
        if crop_region:
            y1, y2, x1, x2 = crop_region
            h, w = frame.shape[:2]

            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                live_crop = frame[y1:y2, x1:x2]
                saved_crop_ref = config_manager.load_cropped_reference(camera_name)

                if saved_crop_ref is not None:
                    saved_resized = cv2.resize(saved_crop_ref, (live_crop.shape[1], live_crop.shape[0]))
                    crop_compare = np.hstack([saved_resized, live_crop])
                    cv2.putText(crop_compare, "Saved", (5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(crop_compare, "Live", (saved_resized.shape[1] + 5, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow("Crop Comparison", crop_compare)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


def print_crop_summary(config_manager: CropConfigManager):
    """Print summary of all saved crop configurations."""
    configs = config_manager.list_configs()

    print(f"\n{'='*60}")
    print(f"  Saved Crop Configurations")
    print(f"{'='*60}")

    if not configs:
        print("\nNo saved configurations found.")
        print(f"Config directory: {config_manager.config_dir}")
        return

    for camera_name, config in configs.items():
        cr = config["crop_region"]
        print(f"\n[{camera_name}]")
        print(f"  Crop: img[{cr['y1']}:{cr['y2']}, {cr['x1']}:{cr['x2']}]")
        print(f"  Size: {config['crop_size']['width']}x{config['crop_size']['height']}")
        print(f"  Image: {config['image_shape']['width']}x{config['image_shape']['height']}")
        print(f"  Updated: {config.get('timestamp', 'unknown')}")
        if config.get('notes'):
            print(f"  Notes: {config['notes']}")

    print(f"\nConfig directory: {config_manager.config_dir}")


def generate_code_snippet(config_manager: CropConfigManager):
    """Generate Python code for all saved crops."""
    configs = config_manager.list_configs()

    if not configs:
        return

    print(f"\n{'='*60}")
    print(f"  Code Snippet (for use in config.py)")
    print(f"{'='*60}")
    print("\nCROP_FUNCTIONS = {")
    for camera_name, config in configs.items():
        cr = config["crop_region"]
        print(f'    "{camera_name}": lambda img: img[{cr["y1"]}:{cr["y2"]}, {cr["x1"]}:{cr["x2"]}],')
    print("}")


def main():
    parser = argparse.ArgumentParser(
        description="Camera crop configuration tool with reference saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--camera', '-c', type=str,
                       choices=list(CAMERA_CONFIG.keys()),
                       help='Camera to configure')
    parser.add_argument('--config-dir', type=str, default=DEFAULT_CONFIG_DIR,
                       help='Directory to store crop configurations')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Select and save crop (default mode)')
    parser.add_argument('--load', '-l', action='store_true',
                       help='Load saved config and adjust if needed')
    parser.add_argument('--compare', action='store_true',
                       help='Compare live camera with saved reference')
    parser.add_argument('--list', action='store_true',
                       help='List all saved configurations')
    parser.add_argument('--delete', action='store_true',
                       help='Delete saved configuration for camera')
    parser.add_argument('--code', action='store_true',
                       help='Generate code snippet for all crops')
    args = parser.parse_args()

    config_manager = CropConfigManager(args.config_dir)

    # List mode
    if args.list:
        print_crop_summary(config_manager)
        if args.code:
            generate_code_snippet(config_manager)
        return

    # Code generation mode
    if args.code and not args.camera:
        generate_code_snippet(config_manager)
        return

    # Camera-specific operations require camera argument
    if not args.camera:
        print("Error: --camera required for this operation")
        print(f"Available cameras: {list(CAMERA_CONFIG.keys())}")
        return

    camera_name = args.camera

    # Delete mode
    if args.delete:
        confirm = input(f"Delete configuration for '{camera_name}'? (y/N): ").strip().lower()
        if confirm == 'y':
            if config_manager.delete_config(camera_name):
                print(f"Configuration deleted for '{camera_name}'")
            else:
                print(f"No configuration found for '{camera_name}'")
        return

    # Start camera
    print(f"Starting camera '{camera_name}'...")
    camera = start_camera(camera_name)
    if camera is None:
        return

    try:
        # Compare mode
        if args.compare:
            compare_with_reference(camera, camera_name, config_manager)
            return

        # Load/adjust mode or Save mode (default)
        initial_crop = None
        if args.load:
            initial_crop = config_manager.get_crop_region(camera_name)
            if initial_crop is None:
                print(f"No saved configuration for '{camera_name}', starting fresh")

        crop_region = select_crop_with_reference(
            camera=camera,
            camera_name=camera_name,
            config_manager=config_manager,
            initial_crop=initial_crop,
        )

        if crop_region:
            y1, y2, x1, x2 = crop_region
            print(f"\n{'='*60}")
            print(f"  Final Crop Configuration")
            print(f"{'='*60}")
            print(f"Camera: {camera_name}")
            print(f"Crop: img[{y1}:{y2}, {x1}:{x2}]")
            print(f"Size: {x2-x1}x{y2-y1}")
            print(f"\nCode snippet:")
            print(f'  "{camera_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}],')
        else:
            print("\nCancelled.")

    finally:
        camera.close()


if __name__ == "__main__":
    main()
