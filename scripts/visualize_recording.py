#!/usr/bin/env python3
"""
Visualize recording .npz files from SessionRecorder.

Supports the new format with:
- timestamps, tactile, wrist_2_images
- ee_poses, actions, episode_ids, intervention_flags, rewards, metadata

Usage:
    python scripts/visualize_recording.py /path/to/recording.npz
    python scripts/visualize_recording.py /path/to/recording.npz --save video.mp4
"""

import sys
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


def load_recording(filepath):
    """Load recording data from .npz file."""
    data = np.load(filepath, allow_pickle=True)

    rec = {
        'timestamps': data['timestamps'],
        'tactile': data['tactile'],
        'wrist_2_images': data['wrist_2_images'],
    }

    # Load new fields if present
    if 'ee_poses' in data:
        rec['ee_poses'] = data['ee_poses']
    if 'actions' in data:
        rec['actions'] = data['actions']
    if 'episode_ids' in data:
        rec['episode_ids'] = data['episode_ids']
    if 'intervention_flags' in data:
        rec['intervention_flags'] = data['intervention_flags']
    if 'rewards' in data:
        rec['rewards'] = data['rewards']
    if 'metadata' in data:
        rec['metadata'] = data['metadata'].item() if data['metadata'].shape == () else data['metadata']

    return rec


def decode_image(img_bytes):
    """Decode JPEG bytes to numpy array."""
    if img_bytes is None or len(img_bytes) == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_recording(filepath, save_path=None):
    """Visualize recording in a single window."""
    print(f"Loading {filepath}...")
    rec = load_recording(filepath)

    n_frames = len(rec['timestamps'])
    duration = rec['timestamps'][-1] if n_frames > 0 else 0
    actual_fps = n_frames / duration if duration > 0 else 0

    # Print metadata
    if 'metadata' in rec:
        meta = rec['metadata']
        print(f"Metadata: {meta}")
    print(f"Loaded {n_frames} frames, duration: {duration:.1f}s, actual FPS: {actual_fps:.1f}")

    # Check for optional fields
    has_ee_poses = 'ee_poses' in rec
    has_actions = 'actions' in rec
    has_episode_ids = 'episode_ids' in rec
    has_intervention = 'intervention_flags' in rec
    has_rewards = 'rewards' in rec

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2, 1, 1, 1])

    # Image subplot (large, top-left)
    ax_img = fig.add_subplot(gs[0, :2])
    ax_img.set_title('Wrist Camera')
    ax_img.axis('off')

    # Info panel (top-right)
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis('off')
    ax_info.set_title('Info')

    # Force plot (row 1)
    ax_force = fig.add_subplot(gs[1, :])
    ax_force.set_title('Tactile Force [Fx, Fy, Fz]')
    ax_force.set_ylabel('Force (N)')
    ax_force.grid(True, alpha=0.3)

    # Torque plot (row 2)
    ax_torque = fig.add_subplot(gs[2, :])
    ax_torque.set_title('Tactile Torque [Mx, My, Mz]')
    ax_torque.set_ylabel('Torque (N·mm)')
    ax_torque.grid(True, alpha=0.3)

    # Position plot (row 3) - if ee_poses available
    ax_pos = fig.add_subplot(gs[3, :])
    ax_pos.set_title('End-Effector Position [X, Y, Z]')
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.grid(True, alpha=0.3)

    # Pre-plot all data
    times = rec['timestamps']
    tactile = rec['tactile']

    # Force lines (Fx, Fy, Fz)
    ax_force.plot(times, tactile[:, 0], 'r-', alpha=0.5, label='Fx')
    ax_force.plot(times, tactile[:, 1], 'g-', alpha=0.5, label='Fy')
    ax_force.plot(times, tactile[:, 2], 'b-', alpha=0.5, label='Fz')
    ax_force.legend(loc='upper right')
    force_vline = ax_force.axvline(x=0, color='k', linestyle='--', linewidth=2)

    # Torque lines (Mx, My, Mz)
    ax_torque.plot(times, tactile[:, 3], 'r-', alpha=0.5, label='Mx')
    ax_torque.plot(times, tactile[:, 4], 'g-', alpha=0.5, label='My')
    ax_torque.plot(times, tactile[:, 5], 'b-', alpha=0.5, label='Mz')
    ax_torque.legend(loc='upper right')
    torque_vline = ax_torque.axvline(x=0, color='k', linestyle='--', linewidth=2)

    # Position lines (X, Y, Z) - if available
    pos_vline = None
    if has_ee_poses:
        ee_poses = rec['ee_poses']
        ax_pos.plot(times, ee_poses[:, 0], 'r-', alpha=0.5, label='X')
        ax_pos.plot(times, ee_poses[:, 1], 'g-', alpha=0.5, label='Y')
        ax_pos.plot(times, ee_poses[:, 2], 'b-', alpha=0.5, label='Z')
        ax_pos.legend(loc='upper right')
        pos_vline = ax_pos.axvline(x=0, color='k', linestyle='--', linewidth=2)
    else:
        ax_pos.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax_pos.transAxes)

    # Initialize image
    first_img = decode_image(rec['wrist_2_images'][0])
    img_display = ax_img.imshow(first_img)

    # Info text
    info_text = ax_info.text(0.05, 0.95, '', fontsize=10, verticalalignment='top',
                             family='monospace', transform=ax_info.transAxes)

    plt.tight_layout()

    # Animation state
    current_frame = [0]
    playing = [True]

    def update(frame_idx):
        """Update function for animation."""
        if frame_idx >= n_frames:
            frame_idx = n_frames - 1

        # Update image
        img = decode_image(rec['wrist_2_images'][frame_idx])
        img_display.set_data(img)

        # Update vertical lines
        t = times[frame_idx]
        force_vline.set_xdata([t, t])
        torque_vline.set_xdata([t, t])
        if pos_vline is not None:
            pos_vline.set_xdata([t, t])

        # Build info string
        fx, fy, fz = tactile[frame_idx, :3]
        mx, my, mz = tactile[frame_idx, 3:6]

        info_str = f"Frame: {frame_idx}/{n_frames-1}\n"
        info_str += f"Time: {t:.2f}s\n\n"

        info_str += f"Force (N):\n"
        info_str += f"  Fx: {fx:+.2f}\n"
        info_str += f"  Fy: {fy:+.2f}\n"
        info_str += f"  Fz: {fz:+.2f}\n\n"

        info_str += f"Torque (N·mm):\n"
        info_str += f"  Mx: {mx:+.2f}\n"
        info_str += f"  My: {my:+.2f}\n"
        info_str += f"  Mz: {mz:+.2f}\n"

        if has_ee_poses:
            x, y, z = rec['ee_poses'][frame_idx, :3]
            info_str += f"\nPosition (m):\n"
            info_str += f"  X: {x:.4f}\n"
            info_str += f"  Y: {y:.4f}\n"
            info_str += f"  Z: {z:.4f}\n"

        if has_actions:
            ax, ay, az = rec['actions'][frame_idx]
            info_str += f"\nAction:\n"
            info_str += f"  [{ax:+.3f}, {ay:+.3f}, {az:+.3f}]\n"

        if has_episode_ids:
            ep_id = rec['episode_ids'][frame_idx]
            info_str += f"\nEpisode: {ep_id}\n"

        if has_intervention:
            is_intvn = rec['intervention_flags'][frame_idx]
            info_str += f"Intervention: {'Yes' if is_intvn else 'No'}\n"

        if has_rewards:
            reward = rec['rewards'][frame_idx]
            if reward != 0:
                info_str += f"Reward: {reward:.1f}\n"

        info_text.set_text(info_str)

        return [img_display, force_vline, torque_vline, info_text]

    if save_path:
        # Save as video - resample to fixed 30fps based on timestamps
        from matplotlib.animation import FFMpegWriter
        print(f"Saving video to {save_path}...")

        # Generate frame indices for 30fps video based on timestamps
        video_fps = 30
        video_duration = duration
        n_video_frames = int(video_duration * video_fps)

        def get_frame_for_time(t):
            """Find the frame index closest to time t."""
            idx = np.searchsorted(times, t)
            if idx >= n_frames:
                idx = n_frames - 1
            return idx

        def update_video(video_frame_idx):
            t = video_frame_idx / video_fps
            frame_idx = get_frame_for_time(t)
            return update(frame_idx)

        writer = FFMpegWriter(fps=video_fps, metadata=dict(artist='SessionRecorder'))
        anim = FuncAnimation(fig, update_video, frames=n_video_frames, interval=1000/video_fps, blit=False)
        anim.save(save_path, writer=writer)
        print(f"Video saved! ({n_video_frames} frames at {video_fps}fps)")
    else:
        # Interactive playback - use real timestamps for correct speed
        print("\nControls:")
        print("  SPACE: Play/Pause")
        print("  LEFT/RIGHT: Step frame")
        print("  HOME/END: Jump to start/end")
        print("  +/-: Speed up/slow down (1x default)")
        print("  Q: Quit")

        playback_speed = [1.0]
        playback_time = [0.0]  # Current playback time in seconds
        last_update_time = [None]

        def get_frame_for_time(t):
            """Find the frame index closest to time t."""
            idx = np.searchsorted(times, t)
            if idx >= n_frames:
                idx = n_frames - 1
            return idx

        def on_key(event):
            """Handle keyboard events."""
            if event.key == ' ':
                playing[0] = not playing[0]
                last_update_time[0] = None  # Reset timing
                print(f"{'Playing' if playing[0] else 'Paused'} (speed: {playback_speed[0]:.1f}x)")
            elif event.key == 'left':
                playback_time[0] = max(0, playback_time[0] - 0.1)
                current_frame[0] = get_frame_for_time(playback_time[0])
                update(current_frame[0])
                fig.canvas.draw_idle()
            elif event.key == 'right':
                playback_time[0] = min(duration, playback_time[0] + 0.1)
                current_frame[0] = get_frame_for_time(playback_time[0])
                update(current_frame[0])
                fig.canvas.draw_idle()
            elif event.key == 'home':
                playback_time[0] = 0
                current_frame[0] = 0
                update(current_frame[0])
                fig.canvas.draw_idle()
            elif event.key == 'end':
                playback_time[0] = duration
                current_frame[0] = n_frames - 1
                update(current_frame[0])
                fig.canvas.draw_idle()
            elif event.key == '+' or event.key == '=':
                playback_speed[0] = min(4.0, playback_speed[0] + 0.5)
                print(f"Speed: {playback_speed[0]:.1f}x")
            elif event.key == '-':
                playback_speed[0] = max(0.25, playback_speed[0] - 0.5)
                print(f"Speed: {playback_speed[0]:.1f}x")
            elif event.key == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)

        import time as time_module

        def animate(_):
            if playing[0]:
                current_time = time_module.time()
                if last_update_time[0] is None:
                    last_update_time[0] = current_time

                # Advance playback time based on real elapsed time
                dt = (current_time - last_update_time[0]) * playback_speed[0]
                last_update_time[0] = current_time

                playback_time[0] += dt
                if playback_time[0] >= duration:
                    playback_time[0] = 0  # Loop

                current_frame[0] = get_frame_for_time(playback_time[0])

            return update(current_frame[0])

        anim = FuncAnimation(fig, animate, interval=33, blit=False, cache_frame_data=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize recording .npz files')
    parser.add_argument('filepath', help='Path to recording .npz file')
    parser.add_argument('--save', '-s', help='Save as video file (e.g., output.mp4)')

    args = parser.parse_args()
    visualize_recording(args.filepath, args.save)


if __name__ == '__main__':
    main()
