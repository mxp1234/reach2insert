#!/usr/bin/env python3
"""
Compare DP collected data with current camera view.

Usage:
    python utils/compare_dp_camera.py

Controls:
    - 'q': quit
    - 'n': next frame
    - 'p': previous frame
    - 'e': next episode
    - 'w': previous episode
    - SPACE: toggle auto-play
    - 's': save comparison screenshot
"""

import cv2
import numpy as np
import sys
import os
import glob
import h5py

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serl_robot_infra", "franka_env"))

from franka_env.camera.rs_capture import RSCapture

# Data path
DATA_PATH = "/home/pi-zero/Documents/openpi/third_party/real_franka/data/peg_in_hole/peg_in_hole_square_III_1-13__mxp"

# Camera config - 640x480 to match recorded data
TOP_CAMERA = {
    "serial_number": "334622072595",
    "dim": (640, 480),
    "exposure": 255,
}


def load_episodes(data_path):
    """Load all episode files."""
    pattern = os.path.join(data_path, "episode_*.hdf5")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} episodes")
    return files


def load_episode_images(filepath, camera_name="top"):
    """Load all images from an episode."""
    images = []
    with h5py.File(filepath, 'r') as f:
        img_data = f['observations']['images'][camera_name]
        for i in range(len(img_data)):
            raw = img_data[i]
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)
    return images


def main():
    print("=" * 60)
    print("DP Data vs Current Camera Comparison")
    print("=" * 60)

    # Load episodes
    episode_files = load_episodes(DATA_PATH)
    if not episode_files:
        print(f"No episodes found in {DATA_PATH}")
        return

    # Initialize camera at 640x480
    print(f"\nInitializing top camera at 640x480...")
    try:
        cap = RSCapture(
            name="top",
            serial_number=TOP_CAMERA["serial_number"],
            dim=TOP_CAMERA["dim"],
            fps=15,
            exposure=TOP_CAMERA["exposure"],
        )
        print(f"  Camera OK (serial: {TOP_CAMERA['serial_number']})")
    except Exception as e:
        print(f"  Camera FAILED: {e}")
        print("\nNote: Make sure no other process is using the camera.")
        return

    # State
    episode_idx = 0
    frame_idx = 0
    auto_play = False
    current_images = None

    def load_current_episode():
        nonlocal current_images, frame_idx
        print(f"\nLoading episode {episode_idx}: {os.path.basename(episode_files[episode_idx])}")
        current_images = load_episode_images(episode_files[episode_idx], "top")
        frame_idx = 0
        print(f"  Loaded {len(current_images)} frames")

    load_current_episode()

    print("\nControls:")
    print("  q: quit | n/p: next/prev frame | e/w: next/prev episode")
    print("  SPACE: toggle auto-play | s: save screenshot")
    print("-" * 60)

    while True:
        # Get current camera frame
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            live_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(live_frame, "Camera Error", (200, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Get recorded frame
        if current_images and 0 <= frame_idx < len(current_images):
            recorded_frame = current_images[frame_idx]
        else:
            recorded_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add labels
        live_display = live_frame.copy()
        recorded_display = recorded_frame.copy()

        cv2.putText(live_display, "LIVE CAMERA (640x480)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(recorded_display, f"RECORDED: ep{episode_idx} frame{frame_idx}/{len(current_images)-1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if auto_play:
            cv2.putText(recorded_display, "AUTO-PLAY", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Side by side comparison
        comparison = np.hstack([recorded_display, live_display])

        # Add divider line
        cv2.line(comparison, (640, 0), (640, 480), (255, 255, 255), 2)

        # Compute difference
        diff = cv2.absdiff(recorded_frame, live_frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_score = np.mean(diff_gray)

        cv2.putText(comparison, f"Diff: {diff_score:.1f}", (600, 470),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show windows
        cv2.imshow("Comparison: RECORDED (left) | LIVE (right)", comparison)
        cv2.imshow("Difference", diff)

        # Handle input
        key = cv2.waitKey(50 if auto_play else 1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('n'):  # Next frame
            frame_idx = min(frame_idx + 1, len(current_images) - 1)
        elif key == ord('p'):  # Previous frame
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('e'):  # Next episode
            episode_idx = min(episode_idx + 1, len(episode_files) - 1)
            load_current_episode()
        elif key == ord('w'):  # Previous episode
            episode_idx = max(episode_idx - 1, 0)
            load_current_episode()
        elif key == ord(' '):  # Toggle auto-play
            auto_play = not auto_play
            print(f"Auto-play: {'ON' if auto_play else 'OFF'}")
        elif key == ord('s'):  # Save screenshot
            filename = f"comparison_ep{episode_idx}_frame{frame_idx}.png"
            cv2.imwrite(filename, comparison)
            print(f"Saved: {filename}")

        # Auto-play advance
        if auto_play:
            frame_idx += 1
            if frame_idx >= len(current_images):
                frame_idx = 0
                episode_idx = (episode_idx + 1) % len(episode_files)
                load_current_episode()

    cap.close()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
