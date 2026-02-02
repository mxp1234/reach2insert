#!/usr/bin/env python3
"""
Demo Recording Script for HIL-SERL

Records demonstrations using SpaceMouse intervention.
Saves successful trajectories to pkl file.

Usage:
    python record_demos.py --exp_name peg_in_hole_square_III --successes_needed 20
"""

import os
import sys
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import select
import termios
import tty

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")


def kbhit():
    """Check if a key was pressed (non-blocking)."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    return dr != []


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, f'Experiment "{FLAGS.exp_name}" not found. Available: {list(CONFIG_MAPPING.keys())}'
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    print(f"=== Demo Recording for {FLAGS.exp_name} ===")
    print(f"Successes needed: {FLAGS.successes_needed}")

    # Create environment
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(__file__), "task", FLAGS.exp_name, "demo_data")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("\nControls:")
    print("  SpaceMouse : Control robot")
    print("  Enter      : End trajectory -> then y/n to judge")
    print("\nStarting in 3 seconds...")
    time.sleep(3.0)

    obs, info = env.reset()
    print("Reset done, start collecting!")

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed, desc="Collecting demos")
    trajectory = []
    traj_count = 0

    # Set terminal to non-blocking mode
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while success_count < success_needed:
            # Check for Enter key (non-blocking)
            if kbhit():
                ch = sys.stdin.read(1)
                if ch == '\n' or ch == '\r':
                    traj_count += 1

                    # Restore terminal for input
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                    print(f"\n[Trajectory {traj_count}] {len(trajectory)} steps")
                    response = input("Success? (y/n): ").strip().lower()

                    # Set back to non-blocking
                    tty.setcbreak(sys.stdin.fileno())

                    if response == 'y' and len(trajectory) > 0:
                        # Mark last transition as terminal
                        trajectory[-1]['rewards'] = 1.0
                        trajectory[-1]['masks'] = 0.0
                        trajectory[-1]['dones'] = True

                        for t in trajectory:
                            transitions.append(copy.deepcopy(t))
                        success_count += 1
                        pbar.update(1)
                        print(f"[SAVED] {len(trajectory)} steps")
                    else:
                        print("[DISCARDED]")

                    trajectory = []
                    print("Resetting...")
                    obs, info = env.reset()
                    print("Ready! Press Enter when done.")
                    continue

            # Step environment
            actions = np.zeros(env.action_space.sample().shape)
            next_obs, rew, done, truncated, info = env.step(actions)

            if "intervene_action" in info:
                actions = info["intervene_action"]

            transition = dict(
                observations=copy.deepcopy(obs),
                actions=actions.copy(),
                next_observations=copy.deepcopy(next_obs),
                rewards=rew,
                masks=1.0,
                dones=False,
                infos={},
            )
            trajectory.append(transition)

            pbar.set_description(f"Traj {traj_count+1} | Steps: {len(trajectory)} | {success_count}/{success_needed}")
            obs = next_obs

            # Handle max episode length
            if done:
                traj_count += 1
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

                print(f"\n[Trajectory {traj_count}] Episode limit ({len(trajectory)} steps)")
                response = input("Success? (y/n): ").strip().lower()

                tty.setcbreak(sys.stdin.fileno())

                if response == 'y' and len(trajectory) > 0:
                    trajectory[-1]['rewards'] = 1.0
                    trajectory[-1]['masks'] = 0.0
                    trajectory[-1]['dones'] = True

                    for t in trajectory:
                        transitions.append(copy.deepcopy(t))
                    success_count += 1
                    pbar.update(1)
                    print(f"[SAVED] {len(trajectory)} steps")
                else:
                    print("[DISCARDED]")

                trajectory = []
                print("Resetting...")
                obs, info = env.reset()
                print("Ready! Press Enter when done.")

    finally:
        # Restore terminal
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    pbar.close()

    # Save
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(output_dir, "demo.pkl")
    backup_name = os.path.join(output_dir, f"demo_{success_needed}_{uuid}.pkl")

    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
    with open(backup_name, "wb") as f:
        pkl.dump(transitions, f)

    print(f"\n=== Done ===")
    print(f"Trajectories: {success_count}")
    print(f"Transitions: {len(transitions)}")
    print(f"Saved: {file_name}")


if __name__ == "__main__":
    app.run(main)
