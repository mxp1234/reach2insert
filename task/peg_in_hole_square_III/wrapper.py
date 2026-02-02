"""
Peg-in-Hole Square III Task Wrappers

Contains:
1. KeyboardRewardWrapper - Manual keyboard success judgement
2. GripperPenaltyWrapper - Gripper action penalty
3. FixedPoseActionWrapper - Fixed orientation and gripper state
4. PegInHoleSquareIIIEnv - Base environment for square peg insertion
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
import time
import sys
import os

# Import base environment
from franka_env.envs.franka_env import FrankaEnv


class KeyboardRewardWrapper(gym.Wrapper):
    """
    Manual keyboard success judgement wrapper

    Replaces automatic reward classifier with human keyboard input:
    - Press '1' or Enter: Success
    - Press '0' or 'n': Failure
    - Press 'r': Request reset (mark as failure and reset)

    Also supports real-time key detection:
    - Press 's': Immediately mark current step as success and end episode
    """

    def __init__(self, env, auto_query=True):
        """
        Args:
            env: Environment to wrap
            auto_query: Whether to auto-query on done (otherwise call query_success manually)
        """
        super().__init__(env)
        self.auto_query = auto_query
        self._success_flag = False
        self._request_reset = False

        # Try using pynput for non-blocking key detection
        try:
            from pynput import keyboard
            self._keyboard_available = True

            def on_press(key):
                try:
                    if hasattr(key, 'char'):
                        if key.char == 's':
                            self._success_flag = True
                            print("\n[KeyboardReward] Success flagged by 's' key")
                        elif key.char == 'r':
                            self._request_reset = True
                            print("\n[KeyboardReward] Reset requested by 'r' key")
                except:
                    pass

            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
        except ImportError:
            self._keyboard_available = False
            print("Warning: pynput not available, using input() for success query")

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)

        # Check for keyboard success flag
        if self._success_flag:
            done = True
            rew = 1
            info['succeed'] = True
            self._success_flag = False
            print("success")
        elif self._request_reset:
            done = True
            rew = 0
            info['succeed'] = False
            self._request_reset = False
            print("reset requested")
        elif done and self.auto_query:
            # Episode ended, query for success
            success = self._query_success()
            rew = 1 if success else 0
            info['succeed'] = success
            if success:
                print("success")

        return obs, rew, done, truncated, info

    def _query_success(self) -> bool:
        """Query user for task success"""
        while True:
            try:
                response = input("\n[KeyboardReward] Task successful? (1/y=yes, 0/n=no): ").strip().lower()
                if response in ['1', 'y', 'yes', '']:
                    print("  -> Marked as SUCCESS")
                    return True
                elif response in ['0', 'n', 'no']:
                    print("  -> Marked as FAILURE")
                    return False
            except KeyboardInterrupt:
                return False

    def reset(self, **kwargs):
        self._success_flag = False
        self._request_reset = False
        return self.env.reset(**kwargs)

    def close(self):
        if self._keyboard_available:
            self._listener.stop()
        super().close()


class GripperPenaltyWrapper(gym.Wrapper):
    """
    Gripper action penalty wrapper
    """

    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0] if "state" in obs else 0.5
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if "intervene_action" in info:
            action = info["intervene_action"]

        # Check for unnecessary gripper actions
        gripper_pos = observation["state"][0, 0] if "state" in observation else 0.5
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or \
           (action[-1] > 0.5 and self.last_gripper_pos < 0.9):
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = gripper_pos
        return observation, reward, terminated, truncated, info


class FixedPoseActionWrapper(gym.Wrapper):
    """
    Fixed pose action wrapper

    Corrects intervene_action to reflect actual executed actions:
    - Rotation action set to 0
    - Gripper action set to closed (-1.0 = closed, 1.0 = open)
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # Correct input action
        action = np.array(action, dtype=np.float32)
        action[3:6] = 0.0  # Zero rotation
        action[6] = -1.0   # Gripper closed

        obs, reward, done, truncated, info = self.env.step(action)

        # Correct intervene_action
        if "intervene_action" in info:
            info["intervene_action"] = np.array(info["intervene_action"], dtype=np.float32)
            info["intervene_action"][3:6] = 0.0
            info["intervene_action"][6] = -1.0  # Gripper closed

        return obs, reward, done, truncated, info


class PegInHoleSquareIIIEnv(FrankaEnv):
    """
    Peg-in-Hole Square III Environment

    Square peg insertion task with:
    - Multi-camera observation
    - Z-range random sampling for reset
    - Fixed orientation control
    """

    def __init__(self, *args, **kwargs):
        self._physical_cameras = None
        self.camera_source = {}
        super().__init__(*args, **kwargs)
        # Get Z sampling range from config
        if hasattr(self.config, 'RESET_Z_RANGE'):
            self.reset_z_range = self.config.RESET_Z_RANGE
        else:
            self.reset_z_range = None
        # Get XY sampling ranges from config (tuple format)
        if hasattr(self.config, 'RESET_X_RANGE'):
            self.reset_x_range = self.config.RESET_X_RANGE
        else:
            self.reset_x_range = None
        if hasattr(self.config, 'RESET_Y_RANGE'):
            self.reset_y_range = self.config.RESET_Y_RANGE
        else:
            self.reset_y_range = None

    def reset(self, **kwargs):
        """
        Peg-in-Hole Square III reset flow

        Steps:
        1. Recover robot to safe state
        2. Switch to precision control mode
        3. Move up 5mm first (safety)
        4. Keep gripper closed, move to reset position
        5. Wait 3 seconds
        6. Move to random start position (if enabled)
        7. Switch to compliance mode
        8. Get initial observation
        """
        import requests
        from franka_env.utils.rotations import euler_2_quat

        # Step 1: Recover robot
        self._recover()
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.1)

        # Step 2: Switch to precision control mode
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.3)

        # Step 3: Move up 5mm first (safety - avoid collision during reset)
        self._update_currpos()
        lift_pose = self.currpos.copy()
        lift_pose[2] += 0.005  # Move up 5mm
        self.interpolate_move(lift_pose, timeout=0.5)
        time.sleep(0.2)

        # Step 4: Keep gripper closed, move to fixed reset position
        requests.post(self.url + "close_gripper")
        time.sleep(0.3)

        reset_pose_fixed = self.resetpos.copy()
        self.interpolate_move(reset_pose_fixed, timeout=1.0)
        time.sleep(0.3)

        # Step 5: Wait 3 seconds
        print("[Reset] Waiting 3 seconds before starting...")
        time.sleep(3.0)

        # Step 6: Calculate random start position and move
        reset_pose = self.resetpos.copy()

        if self.randomreset:
            # XY randomization - use tuple ranges if available, else scalar offset
            if self.reset_x_range is not None and self.reset_y_range is not None:
                # Sample X and Y independently from their ranges
                reset_pose[0] = np.random.uniform(self.reset_x_range[0], self.reset_x_range[1])
                reset_pose[1] = np.random.uniform(self.reset_y_range[0], self.reset_y_range[1])
            elif hasattr(self, 'random_xy_range') and isinstance(self.random_xy_range, (int, float)):
                # Fallback to scalar offset
                reset_pose[:2] += np.random.uniform(
                    -self.random_xy_range, self.random_xy_range, (2,)
                )

            # Z range random sampling
            if self.reset_z_range is not None:
                z_min, z_max = self.reset_z_range
                reset_pose[2] = np.random.uniform(z_min, z_max)

            # Rotation randomization (if needed)
            if self.random_rz_range > 0:
                euler_random = self._RESET_POSE[3:].copy()
                euler_random[-1] += np.random.uniform(
                    -self.random_rz_range, self.random_rz_range
                )
                reset_pose[3:] = euler_2_quat(euler_random)

        # Move to random start position
        self.interpolate_move(reset_pose, timeout=1.0)
        time.sleep(0.3)

        # Step 7: Switch to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        time.sleep(0.1)

        # Step 8: Get initial observation
        self.curr_path_length = 0
        self._update_currpos()
        obs = self._get_obs()

        return obs, {}

    def init_cameras(self, name_serial_dict=None):
        """
        Initialize multiple cameras - handle shared cameras

        top_crop and side share the same physical camera, only open once
        """
        import time
        from collections import OrderedDict
        from franka_env.camera.video_capture import VideoCapture
        from franka_env.camera.rs_capture import RSCapture

        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        self.camera_source = {}  # cam_name -> source_cam_name
        self._physical_cameras = OrderedDict()  # Only store physical cameras
        serial_to_cam = {}  # serial -> first_cam_name

        for cam_name, kwargs in name_serial_dict.items():
            serial = kwargs.get('serial_number', cam_name)

            if serial in serial_to_cam:
                # Physical camera already open, reuse
                source_cam = serial_to_cam[serial]
                self.cap[cam_name] = self._physical_cameras[source_cam]
                self.camera_source[cam_name] = source_cam
                print(f"  {cam_name}: reusing {source_cam}")
            else:
                # New physical camera
                cap = VideoCapture(
                    RSCapture(name=cam_name, **kwargs)
                )
                self.cap[cam_name] = cap
                self._physical_cameras[cam_name] = cap
                serial_to_cam[serial] = cam_name
                self.camera_source[cam_name] = cam_name
                print(f"  {cam_name}: {serial}")
                time.sleep(0.5)  # Allow camera to warm up

    def get_im(self):
        """
        Get images - handle shared cameras

        Read once from each physical camera, then apply corresponding crop for each cam_name
        """
        import queue
        import cv2
        import copy

        # 1. Read once from each physical camera
        raw_images = {}
        for cam_name, cap in self._physical_cameras.items():
            try:
                raw_images[cam_name] = cap.read()
            except queue.Empty:
                input(f"{cam_name} camera frozen. Check connect, then press enter...")
                cap.close()
                self.init_cameras(self.config.REALSENSE_CAMERAS)
                return self.get_im()

        # 2. Process image for each cam_name
        images = {}
        display_images = {}
        full_res_images = {}

        for key in self.cap.keys():
            source_cam = self.camera_source.get(key, key)
            rgb = raw_images[source_cam].copy()  # Use copy to avoid modifying original

            # Apply cam_name specific crop
            cropped_rgb = self.config.IMAGE_CROP[key](rgb) if key in self.config.IMAGE_CROP else rgb

            # Resize
            resized = cv2.resize(
                cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = resized[..., ::-1]  # BGR to RGB
            display_images[key] = resized
            display_images[key + "_full"] = cropped_rgb
            full_res_images[key] = copy.deepcopy(cropped_rgb)

        # Save video
        if self.save_video:
            self.recording_frames.append(full_res_images)

        # Display images
        if self.display_image:
            self.img_queue.put(display_images)

        return images

    def close_cameras(self):
        """Close cameras - only close physical cameras to avoid duplicate close"""
        try:
            if self._physical_cameras is not None:
                for cap in self._physical_cameras.values():
                    cap.close()
                self._physical_cameras = None
        except Exception as e:
            print(f"Failed to close cameras: {e}")

    def go_to_reset(self, joint_reset=False):
        """
        Override reset logic, support Z-range random sampling

        XY randomized around RESET_POSE (controlled by RANDOM_XY_RANGE)
        Z randomly sampled within RESET_Z_RANGE (above hole area)
        """
        import requests
        from franka_env.utils.rotations import euler_2_quat

        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # Calculate reset position
        reset_pose = self.resetpos.copy()

        if self.randomreset:
            # XY randomization (relative to RESET_POSE)
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )

            # Z range random sampling (absolute range)
            if self.reset_z_range is not None:
                z_min, z_max = self.reset_z_range
                reset_pose[2] = np.random.uniform(z_min, z_max)

            # Rotation randomization (if needed)
            if self.random_rz_range > 0:
                euler_random = self._RESET_POSE[3:].copy()
                euler_random[-1] += np.random.uniform(
                    -self.random_rz_range, self.random_rz_range
                )
                reset_pose[3:] = euler_2_quat(euler_random)

        self.interpolate_move(reset_pose, timeout=1)

        # Switch to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)

    def step(self, action):
        """
        Override step - force lock orientation and gripper

        1. Only use xyz translation
        2. Force orientation to [pi, 0, 0]
        3. Force gripper to stay closed
        """
        from scipy.spatial.transform import Rotation
        from franka_env.utils.rotations import euler_2_quat

        start_time = time.time()

        # Process action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        xyz_delta = action[:3]

        # Calculate next position - only update xyz
        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta

        # Force orientation to [pi, 0, 0]
        target_quat = euler_2_quat(np.array([np.pi, 0, 0]))
        self.nextpos[3:] = target_quat

        # Force gripper closed (send negative close command)
        self._send_gripper_command(-1.0)

        # Send position command
        self._send_pos_command(self.clip_safety_box(self.nextpos))

        # Update counter and wait
        self.curr_path_length += 1
        dt = time.time() - start_time
        time.sleep(max(0, (1.0 / self.hz) - dt))

        # Get observation
        self._update_currpos()
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate

        return obs, int(reward), done, False, {"succeed": reward}
