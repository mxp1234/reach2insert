#!/usr/bin/env python3
"""
Camera Viewer - View selected cameras for verification

Usage:
    python utils/view_cameras.py
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "serl_robot_infra", "franka_env"))

from franka_env.camera.rs_capture import RSCapture

# Camera config
CAMERAS = {
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
}

# Crop region for top camera: img[y1:y2, x1:x2]
TOP_CROP = (153,212, 348,407)  # y1, y2, x1, x2 (scaled from 640x480 to 1280x720)


def main():
    print("Starting cameras...")

    caps = {}
    for name, cfg in CAMERAS.items():
        try:
            cap = RSCapture(
                name=name,
                serial_number=cfg["serial_number"],
                dim=cfg["dim"],
                fps=15,
                exposure=cfg["exposure"],
            )
            caps[name] = cap
            print(f"  {name}: OK (serial: {cfg['serial_number']})")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    if not caps:
        print("No cameras available!")
        return

    print(f"\nTop crop region: y=[{TOP_CROP[0]}:{TOP_CROP[1]}], x=[{TOP_CROP[2]}:{TOP_CROP[3]}]")
    print("Press 'q' to quit, 's' to save screenshot")

    while True:
        # wrist_2
        if "wrist_2" in caps:
            ret, frame = caps["wrist_2"].read()
            if ret and frame is not None:
                cv2.imshow("wrist_2 (full)", frame)
                resized = cv2.resize(frame, (128, 128))
                cv2.imshow("wrist_2 (128x128)", resized)

        # top camera - show full frame with crop box
        if "top" in caps:
            ret, frame = caps["top"].read()
            if ret and frame is not None:
                y1, y2, x1, x2 = TOP_CROP

                # Full frame with crop rectangle
                full_with_box = frame.copy()
                cv2.rectangle(full_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(full_with_box, f"Crop: [{y1}:{y2}, {x1}:{x2}]", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("top (full + crop box)", full_with_box)

                # Cropped region
                cropped = frame[y1:y2, x1:x2]
                cv2.imshow("top (cropped)", cropped)

                # Resized to 128x128
                resized = cv2.resize(cropped, (128, 128))
                cv2.imshow("top (128x128)", resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if "top" in caps:
                ret, frame = caps["top"].read()
                if ret:
                    cv2.imwrite("top_full.png", frame)
                    y1, y2, x1, x2 = TOP_CROP
                    cv2.imwrite("top_cropped.png", frame[y1:y2, x1:x2])
                    print("Saved: top_full.png, top_cropped.png")

    for cap in caps.values():
        cap.close()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
