#!/usr/bin/env python3
"""
Visualize HIL-SERL buffer images

Usage:
    python utils/visualize_buffer.py --buffer task/peg_in_hole_square_III/checkpoints/buffer/transitions_2000.pkl --camera top
"""

import os
import sys
import argparse
import pickle
import numpy as np
import cv2

def load_buffer_images(buffer_path: str, camera_name: str, max_images: int = 100):
    """Load images from HIL-SERL buffer pickle file."""
    print(f"Loading buffer: {buffer_path}")

    with open(buffer_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Total transitions: {len(data)}")

    images = []
    step = max(1, len(data) // max_images)

    for i in range(0, len(data), step):
        if len(images) >= max_images:
            break

        transition = data[i]
        obs = transition.get('observations', {})

        if camera_name in obs:
            img = obs[camera_name]
            # Shape is (1, 128, 128, 3)
            if img.ndim == 4:
                img = img[0]

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            # Convert RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)

    print(f"Loaded {len(images)} images for camera '{camera_name}'")
    return images


def visualize_buffer(images, camera_name, window_name="Buffer Viewer"):
    """Visualize buffer images."""
    if not images:
        print("No images to display")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 512, 512)

    idx = 0
    n_images = len(images)
    h, w = images[0].shape[:2]

    print(f"\n=== Buffer Viewer for '{camera_name}' ===")
    print(f"Image size: {w}x{h}")
    print(f"Total images: {n_images}")
    print("\nControls:")
    print("  A/D or Left/Right : Navigate images")
    print("  Q/Escape          : Quit")

    while True:
        frame = images[idx].copy()

        # Resize for display
        display = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)

        # Add info
        info_text = f"[{idx+1}/{n_images}] {camera_name} ({w}x{h})"
        cv2.putText(display, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('a') or key == 81:  # A or Left
            idx = (idx - 1) % n_images
        elif key == ord('d') or key == 83:  # D or Right
            idx = (idx + 1) % n_images
        elif key == ord('q') or key == 27:  # Q or Escape
            break

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize HIL-SERL buffer images")
    parser.add_argument('--buffer', '-b', type=str, required=True,
                        help='Path to buffer pickle file')
    parser.add_argument('--camera', '-c', type=str, default='top',
                        help='Camera name to visualize (top, side, wrist_2)')
    parser.add_argument('--max-images', '-n', type=int, default=100,
                        help='Maximum number of images to load')
    args = parser.parse_args()

    if not os.path.exists(args.buffer):
        print(f"Error: Buffer file not found: {args.buffer}")
        return

    images = load_buffer_images(args.buffer, args.camera, args.max_images)

    if images:
        visualize_buffer(images, args.camera)


if __name__ == "__main__":
    main()
