#!/usr/bin/env python3
"""
Interactive Image Crop Selector

Allows manual selection of crop regions for camera images.
Supports both live camera and dataset visualization.

Usage:
    # Live camera mode (1280x720)
    python utils/crop_selector.py --camera top

    # Live camera with preset crop (verify if camera position changed)
    python utils/crop_selector.py --camera top --crop "214:400,636:799"

    # Dataset visualization mode
    python utils/crop_selector.py --dataset /path/to/dataset --camera top

    # Manual crop input mode
    python utils/crop_selector.py --dataset /path/to/dataset --camera top --crop "214:400,636:799"

    # Compare current camera with dataset
    python utils/crop_selector.py --compare --dataset /path/to/dataset --camera top

Controls:
    Mouse drag  : Select crop region
    C           : Input crop manually (e.g., "214:400,636:799")
    R           : Reset selection
    Enter/Space : Confirm selection
    Q/Escape    : Cancel
"""

import os
import sys
import argparse
import numpy as np
import cv2
import h5py
import glob
from typing import Optional, Tuple, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Global variables for mouse callback
drawing = False
start_point = None
end_point = None
current_rect = None
selected_rect = None

# Camera configurations (1280x720)
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
    # Remove brackets and spaces
    crop_str = crop_str.strip().strip('[]').replace(' ', '')

    parts = crop_str.split(',')
    if len(parts) != 2:
        raise ValueError(f"Invalid crop format: {crop_str}. Expected 'y1:y2,x1:x2'")

    y_part, x_part = parts
    y1, y2 = map(int, y_part.split(':'))
    x1, x2 = map(int, x_part.split(':'))

    return (y1, y2, x1, x2)


def load_dataset_images(dataset_path: str, camera_name: str, max_images: int = 100) -> List[np.ndarray]:
    """
    Load images from HDF5 dataset files.

    Args:
        dataset_path: Path to dataset directory containing .hdf5 files
        camera_name: Name of camera (e.g., 'top', 'wrist_2')
        max_images: Maximum number of images to load

    Returns:
        List of images as numpy arrays
    """
    images = []

    # Find all HDF5 files
    hdf5_files = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")))

    if not hdf5_files:
        print(f"No HDF5 files found in {dataset_path}")
        return images

    print(f"Found {len(hdf5_files)} HDF5 files")

    for hdf5_file in hdf5_files:
        if len(images) >= max_images:
            break

        try:
            with h5py.File(hdf5_file, 'r') as f:
                # Check if images exist
                img_path = f"observations/images/{camera_name}"
                if img_path not in f:
                    # Try alternative paths
                    alt_paths = [
                        f"observations/{camera_name}",
                        f"images/{camera_name}",
                        camera_name,
                    ]
                    found = False
                    for alt_path in alt_paths:
                        if alt_path in f:
                            img_path = alt_path
                            found = True
                            break
                    if not found:
                        continue

                # Get compression flag
                compress_images = f.attrs.get("compress_images", True)

                # Load images
                img_data = f[img_path]
                n_frames = min(len(img_data), max_images - len(images))

                for i in range(0, n_frames, max(1, n_frames // 10)):  # Sample evenly
                    if len(images) >= max_images:
                        break

                    if compress_images:
                        # Decompress JPEG
                        compressed = img_data[i]
                        img = cv2.imdecode(
                            np.frombuffer(compressed, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                    else:
                        img = img_data[i]
                        if img.shape[-1] == 3 and img.dtype == np.uint8:
                            # Might be RGB, convert to BGR for OpenCV
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    if img is not None:
                        images.append(img)

        except Exception as e:
            print(f"Error loading {hdf5_file}: {e}")
            continue

    print(f"Loaded {len(images)} images for camera '{camera_name}'")
    return images


def visualize_dataset_with_crop(
    images: List[np.ndarray],
    camera_name: str,
    crop_region: Optional[Tuple[int, int, int, int]] = None,
    window_name: str = "Dataset Viewer"
):
    """
    Visualize dataset images with optional crop overlay.

    Controls:
    - Left/Right arrows or A/D: Navigate images
    - R: Reset to first image
    - C: Input new crop region
    - Enter: Confirm and exit
    - Q/Escape: Cancel
    """
    global selected_rect, current_rect

    if not images:
        print("No images to display")
        return None

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Initialize crop region
    if crop_region:
        y1, y2, x1, x2 = crop_region
        selected_rect = ((x1, y1), (x2, y2))
        current_rect = selected_rect
    else:
        selected_rect = None
        current_rect = None

    idx = 0
    n_images = len(images)
    h, w = images[0].shape[:2]

    print(f"\n=== Dataset Viewer for '{camera_name}' ===")
    print(f"Image size: {w}x{h}")
    print(f"Total images: {n_images}")
    print("\nControls:")
    print("  A/D or Left/Right : Navigate images")
    print("  Mouse drag        : Select new crop region")
    print("  C                 : Input crop range manually")
    print("  R                 : Reset selection")
    print("  Enter/Space       : Confirm selection")
    print("  Q/Escape          : Cancel")

    if crop_region:
        y1, y2, x1, x2 = crop_region
        print(f"\nCurrent crop: img[{y1}:{y2}, {x1}:{x2}]")

    while True:
        frame = images[idx]
        display = frame.copy()

        # Draw current/selected rectangle
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

        # Show info
        info_text = f"[{idx+1}/{n_images}] Size: {w}x{h}"
        cv2.putText(display, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if selected_rect:
            p1, p2 = selected_rect
            crop_text = f"Crop: img[{p1[1]}:{p2[1]}, {p1[0]}:{p2[0]}]"
            cv2.putText(display, crop_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        # Show cropped preview
        if selected_rect:
            p1, p2 = selected_rect
            y1, y2 = p1[1], p2[1]
            x1, x2 = p1[0], p2[0]
            if 0 <= y1 < y2 <= h and 0 <= x1 < x2 <= w:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size > 0:
                    cv2.imshow("Cropped Preview", cropped)
                    resized = cv2.resize(cropped, (128, 128))
                    cv2.imshow("128x128 Preview", resized)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('a') or key == 81:  # A or Left arrow
            idx = (idx - 1) % n_images
        elif key == ord('d') or key == 83:  # D or Right arrow
            idx = (idx + 1) % n_images
        elif key == ord('r'):  # Reset
            current_rect = None
            selected_rect = None
            try:
                cv2.destroyWindow("Cropped Preview")
                cv2.destroyWindow("128x128 Preview")
            except:
                pass
            print("Selection reset")
        elif key == ord('c'):  # Manual crop input
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
        elif key == 13 or key == 32:  # Enter or Space
            if selected_rect:
                cv2.destroyAllWindows()
                p1, p2 = selected_rect
                return (p1[1], p2[1], p1[0], p2[0])  # y1, y2, x1, x2
            else:
                print("No region selected!")
        elif key == ord('q') or key == 27:  # Q or Escape
            cv2.destroyAllWindows()
            return None

    return None


def compare_camera_with_dataset(
    dataset_path: str,
    camera_name: str,
    crop_region: Optional[Tuple[int, int, int, int]] = None
):
    """
    Compare live camera view with dataset images side by side.
    Useful for checking if camera position has changed.
    """
    # Try to import camera
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "serl_robot_infra", "franka_env"))
        from franka_env.camera.rs_capture import RSCapture
        camera_available = True
    except ImportError:
        print("Warning: Camera module not available")
        camera_available = False

    # Load dataset images
    dataset_images = load_dataset_images(dataset_path, camera_name, max_images=50)
    if not dataset_images:
        print("No dataset images found")
        return

    # Start camera
    camera = None
    if camera_available:
        cfg = CAMERA_CONFIG.get(camera_name)
        if cfg:
            try:
                camera = RSCapture(
                    name=camera_name,
                    serial_number=cfg["serial_number"],
                    dim=cfg["dim"],
                    fps=15,
                    exposure=cfg["exposure"],
                )
                print(f"Camera '{camera_name}' started")
            except Exception as e:
                print(f"Failed to start camera: {e}")
                camera = None

    dataset_idx = 0
    n_dataset = len(dataset_images)

    print(f"\n=== Camera vs Dataset Comparison ===")
    print(f"Dataset images: {n_dataset}")
    print("\nControls:")
    print("  A/D: Navigate dataset images")
    print("  Q: Quit")

    window_name = "Comparison (Left: Dataset, Right: Live Camera)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        # Get dataset image
        dataset_frame = dataset_images[dataset_idx]

        # Get live camera image (or use placeholder)
        if camera:
            ret, live_frame = camera.read()
            if not ret:
                live_frame = np.zeros_like(dataset_frame)
                cv2.putText(live_frame, "Camera Error", (50, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            live_frame = np.zeros_like(dataset_frame)
            cv2.putText(live_frame, "Camera Not Available", (50, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Resize to same size if needed
        h1, w1 = dataset_frame.shape[:2]
        h2, w2 = live_frame.shape[:2]
        if (h1, w1) != (h2, w2):
            live_frame = cv2.resize(live_frame, (w1, h1))

        # Draw crop region on both
        if crop_region:
            y1, y2, x1, x2 = crop_region
            cv2.rectangle(dataset_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(live_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add labels
        cv2.putText(dataset_frame, f"Dataset [{dataset_idx+1}/{n_dataset}]", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(live_frame, "Live Camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Combine side by side
        combined = np.hstack([dataset_frame, live_frame])

        cv2.imshow(window_name, combined)

        # Show cropped previews
        if crop_region:
            y1, y2, x1, x2 = crop_region
            dataset_crop = dataset_images[dataset_idx][y1:y2, x1:x2]
            cv2.imshow("Dataset Crop", dataset_crop)

            if camera:
                ret, live = camera.read()
                if ret:
                    live_crop = live[y1:y2, x1:x2]
                    cv2.imshow("Live Crop", live_crop)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('a') or key == 81:
            dataset_idx = (dataset_idx - 1) % n_dataset
        elif key == ord('d') or key == 83:
            dataset_idx = (dataset_idx + 1) % n_dataset
        elif key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()
    if camera:
        camera.close()


def select_crop_region(
    camera,
    camera_name: str,
    initial_crop: Optional[Tuple[int, int, int, int]] = None,
    window_name: str = "Crop Selector"
) -> Optional[Tuple[int, int, int, int]]:
    """
    Interactive crop region selection from live camera.

    Args:
        camera: Camera object with read() method
        camera_name: Name of camera
        initial_crop: Optional initial crop region (y1, y2, x1, x2)
        window_name: Window title
    """
    global drawing, start_point, end_point, current_rect, selected_rect

    drawing = False
    start_point = None
    end_point = None
    current_rect = None
    selected_rect = None

    # Initialize with provided crop region
    if initial_crop:
        y1, y2, x1, x2 = initial_crop
        selected_rect = ((x1, y1), (x2, y2))
        current_rect = selected_rect

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    ret, frame = camera.read()
    if not ret:
        print("Failed to read from camera")
        return None

    h, w = frame.shape[:2]

    print(f"\n=== Crop Selector for '{camera_name}' ===")
    print(f"Image size: {w}x{h}")
    if initial_crop:
        y1, y2, x1, x2 = initial_crop
        print(f"Initial crop: img[{y1}:{y2}, {x1}:{x2}]")
    print("\nControls:")
    print("  Mouse drag  : Select crop region")
    print("  C           : Input crop manually")
    print("  R           : Reset selection")
    print("  Enter/Space : Confirm selection")
    print("  Q/Escape    : Cancel")

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        display = frame.copy()

        if current_rect or selected_rect:
            rect = current_rect if current_rect else selected_rect
            p1, p2 = rect
            cv2.rectangle(display, p1, p2, (0, 255, 0), 2)

            w_rect = abs(p2[0] - p1[0])
            h_rect = abs(p2[1] - p1[1])
            text = f"{w_rect}x{h_rect}"
            cv2.putText(display, text, (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        info_text = f"Size: {w}x{h}"
        cv2.putText(display, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if selected_rect:
            p1, p2 = selected_rect
            crop_text = f"Crop: img[{p1[1]}:{p2[1]}, {p1[0]}:{p2[0]}]"
            cv2.putText(display, crop_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        if selected_rect:
            p1, p2 = selected_rect
            cropped = frame[p1[1]:p2[1], p1[0]:p2[0]]
            if cropped.size > 0:
                cv2.imshow("Cropped Preview", cropped)
                resized = cv2.resize(cropped, (128, 128))
                cv2.imshow("128x128 Preview", resized)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('r'):
            current_rect = None
            selected_rect = None
            try:
                cv2.destroyWindow("Cropped Preview")
                cv2.destroyWindow("128x128 Preview")
            except:
                pass
            print("Selection reset")
        elif key == ord('c'):  # Manual crop input
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
        elif key == 13 or key == 32:
            if selected_rect:
                cv2.destroyAllWindows()
                p1, p2 = selected_rect
                return (p1[1], p2[1], p1[0], p2[0])
            else:
                print("No region selected!")
        elif key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return None

    return None


def generate_crop_config(
    camera_name: str,
    crop_region: Tuple[int, int, int, int],
    image_shape: Tuple[int, int]
) -> str:
    """Generate the config code for the crop."""
    y1, y2, x1, x2 = crop_region
    crop_lambda = f'"{camera_name}": lambda img: img[{y1}:{y2}, {x1}:{x2}]'
    return crop_lambda


def main():
    parser = argparse.ArgumentParser(description="Interactive crop region selector")
    parser.add_argument('--camera', '-c', default='top',
                        choices=list(CAMERA_CONFIG.keys()),
                        help='Camera to select crop for')
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Path to dataset directory (for visualization mode)')
    parser.add_argument('--crop', type=str, default=None,
                        help='Manual crop range, e.g., "214:400,636:799"')
    parser.add_argument('--compare', action='store_true',
                        help='Compare live camera with dataset')
    args = parser.parse_args()

    camera_name = args.camera
    cfg = CAMERA_CONFIG[camera_name]

    # Parse crop if provided
    crop_region = None
    if args.crop:
        try:
            crop_region = parse_crop_string(args.crop)
            y1, y2, x1, x2 = crop_region
            print(f"Using crop: img[{y1}:{y2}, {x1}:{x2}]")
        except Exception as e:
            print(f"Error parsing crop: {e}")
            return

    # Mode 1: Compare camera with dataset
    if args.compare:
        if not args.dataset:
            print("Error: --dataset required for compare mode")
            return
        compare_camera_with_dataset(args.dataset, camera_name, crop_region)
        return

    # Mode 2: Dataset visualization
    if args.dataset:
        print(f"Loading dataset from: {args.dataset}")
        images = load_dataset_images(args.dataset, camera_name)

        if not images:
            print("No images loaded")
            return

        crop_region = visualize_dataset_with_crop(images, camera_name, crop_region)

        if crop_region is None:
            print("\nSelection cancelled.")
            return

        y1, y2, x1, x2 = crop_region
        print(f"\n=== Selected Crop Region ===")
        print(f"  Y range: [{y1}, {y2}] (height: {y2 - y1})")
        print(f"  X range: [{x1}, {x2}] (width: {x2 - x1})")

        config_code = generate_crop_config(camera_name, crop_region, cfg["dim"])
        print(f"\n=== Config Code (for config.py) ===")
        print(f"  {config_code},")
        print(f"\n=== Direct Slice ===")
        print(f"  img[{y1}:{y2}, {x1}:{x2}]")
        return

    # Mode 3: Live camera mode
    print(f"Starting camera '{camera_name}'...")
    print(f"  Serial: {cfg['serial_number']}")
    print(f"  Resolution: {cfg['dim']}")
    if crop_region:
        y1, y2, x1, x2 = crop_region
        print(f"  Initial crop: img[{y1}:{y2}, {x1}:{x2}]")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "serl_robot_infra", "franka_env"))
        from franka_env.camera.rs_capture import RSCapture

        camera = RSCapture(
            name=camera_name,
            serial_number=cfg["serial_number"],
            dim=cfg["dim"],
            fps=15,
            exposure=cfg["exposure"],
        )
    except Exception as e:
        print(f"Error: Failed to start camera: {e}")
        return

    crop_region = select_crop_region(camera, camera_name, initial_crop=crop_region)
    camera.close()

    if crop_region is None:
        print("\nSelection cancelled.")
        return

    y1, y2, x1, x2 = crop_region
    print(f"\n=== Selected Crop Region ===")
    print(f"  Y range: [{y1}, {y2}] (height: {y2 - y1})")
    print(f"  X range: [{x1}, {x2}] (width: {x2 - x1})")

    config_code = generate_crop_config(camera_name, crop_region, cfg["dim"])

    print(f"\n=== Config Code (for config.py) ===")
    print(f"  {config_code},")

    print(f"\n=== Direct Slice ===")
    print(f"  img[{y1}:{y2}, {x1}:{x2}]")


if __name__ == "__main__":
    main()
