#!/usr/bin/env python3
"""
SpaceMouse Robot Control Test

Uses the working SpaceMouseIntervention class from diffusion_policy
Reference: /home/pi-zero/Documents/diffusion_policy/scripts/eval_franka_intervention.py
"""

import time
import numpy as np
import requests
import sys

# Add diffusion_policy to path
sys.path.insert(0, "/home/pi-zero/Documents/diffusion_policy")

from spacemouse import SpaceMouseIntervention

# Robot configuration
ROBOT_URL = "http://172.16.0.1:5000"


def get_robot_state(robot_url):
    try:
        response = requests.post(f"{robot_url}/getstate", timeout=0.5)
        if response.status_code == 200:
            s = response.json()
            return {
                "ee_6d": np.array(s["ee"], dtype=np.float32),
                "gripper_pos": float(s.get("gripper_pos", 0.0)),
            }
    except Exception as e:
        print(f"Error getting robot state: {e}")
    return None


def send_action(robot_url, pose):
    try:
        requests.post(f"{robot_url}/pose", json={"arr": pose.tolist()}, timeout=0.5)
    except Exception as e:
        print(f"Error sending action: {e}")


def close_gripper(robot_url):
    try:
        requests.post(f"{robot_url}/close_gripper", timeout=0.5)
    except:
        pass


def main():
    print("=" * 60)
    print("  SpaceMouse Robot Control Test (Using Verified Implementation)")
    print("=" * 60)
    print("Move the SpaceMouse to control the robot.")
    print("Press Ctrl+C to stop.\n")

    # Connect to robot
    print("[Init] Connecting to robot...")
    robot_state = get_robot_state(ROBOT_URL)
    if robot_state is None:
        print("ERROR: Cannot connect to robot!")
        return
    print(f"  Robot OK. Position: {robot_state['ee_6d'][:3]}")

    # Initialize SpaceMouseIntervention (uses the verified working implementation)
    print("[Init] Opening SpaceMouse...")
    intervention = SpaceMouseIntervention(
        spacemouse_scale=0.05,  # Verified working scale from eval_franka_intervention.py
        policy_scale=0.015,
        rotation_scale=1.0,
        gripper_enabled=False,  # Gripper disabled for peg-in-hole
        intervention_threshold=0.001,
        action_dim=7,
    )
    print("  SpaceMouse OK")

    # Close gripper
    close_gripper(ROBOT_URL)

    print("\n" + "=" * 60)
    print("Ready! Move SpaceMouse to control robot.")
    print("  - Push/Pull: Forward/Backward (X)")
    print("  - Left/Right: Left/Right (Y)")
    print("  - Up/Down: Up/Down (Z)")
    print("=" * 60 + "\n")

    control_hz = 10.0
    dt = 1.0 / control_hz

    try:
        while True:
            t_start = time.time()

            # Get current robot state
            robot_state = get_robot_state(ROBOT_URL)
            if robot_state is None:
                continue

            current_pose = robot_state["ee_6d"].copy()

            # Create zero policy action (spacemouse will override)
            policy_action = np.zeros(7)

            # Get SpaceMouse action using the intervention class
            # This automatically handles scaling and deadzone
            final_action, was_intervened, info = intervention.get_action(
                policy_action, scale_policy=False
            )

            if was_intervened:
                # Apply delta to current pose
                target_pose = current_pose.copy()
                target_pose[0] += final_action[0]  # X
                target_pose[1] += final_action[1]  # Y
                target_pose[2] -= final_action[2]  # Z (inverted)
                # Keep orientation fixed (indices 3-6 are quaternion)

                # Send command
                send_action(ROBOT_URL, target_pose)

                print(f"\r[MOVING] Delta: [{final_action[0]*1000:+.2f}, {final_action[1]*1000:+.2f}, {final_action[2]*1000:+.2f}] mm | "
                      f"Pos: [{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]", end='')
            else:
                pos = current_pose[:3]
                print(f"\r[IDLE  ] Delta: [+0.00, +0.00, +0.00] mm | "
                      f"Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]", end='')

            # Wait for next control step
            elapsed = time.time() - t_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        intervention.close()
        print("SpaceMouse closed.")


if __name__ == "__main__":
    main()
