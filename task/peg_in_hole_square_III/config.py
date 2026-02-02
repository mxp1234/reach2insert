"""
Peg-in-Hole Square III Task Configuration

Square peg insertion task with:
1. Visual observation from cameras
2. Keyboard-based reward (manual success judgement)
3. Fixed gripper (always closed)
4. For the second stage of DP+SERL pipeline

When recording new demos, set MANUAL_POSE_CONFIG = True and configure the poses manually.
After collecting demos, set MANUAL_POSE_CONFIG = False to auto-compute from demo data.
"""

import os
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from task.config import DefaultTrainingConfig
from task.peg_in_hole_square_III.wrapper import (
    PegInHoleSquareIIIEnv,
    KeyboardRewardWrapper,
    GripperPenaltyWrapper,
    FixedPoseActionWrapper,
)


# ============================================
# Configuration Mode
# Set to True when recording new demos (uses manual poses)
# Set to False after demos are collected (auto-computes from demo data)
# ============================================
MANUAL_POSE_CONFIG = True


# ============================================
# Peg-in-Hole Parameters
# ============================================
PEG_LENGTH = 0.05       # 5cm peg length
ROTATION_MARGIN = 0.1   # Rotation tolerance (radians)

# Separate X/Y margins (meters) - (lower_offset, upper_offset) from KDE mode
X_MARGIN = (0.002, 0.01)  # (lower, upper) margins for X axis
Y_MARGIN = (0.006, 0.006)  # (lower, upper) margins for Y axis


if MANUAL_POSE_CONFIG:
    # ============================================
    # Manual Pose Configuration (for demo recording)
    # TODO: Update these values based on current object position
    # ============================================

    # Manual reset pose: where robot starts each episode
    # Move robot to desired start position and record the pose here
    _RESET_POSE = np.array([
        0.574,      # X position (meters)
        -0.090,     # Y position (meters)
        0.20,       # Z position (meters) - start height above insertion
        np.pi, 0, 0 # Orientation: [pi, 0, 0] pointing down
    ])

    # Manual target pose: insertion goal position
    # This is where the peg should be fully inserted
    _TARGET_POSE = np.array([
        0.574,      # X position (same as reset)
        -0.090,     # Y position (same as reset)
        0.145,      # Z position (meters) - insertion depth
        np.pi, 0, 0 # Orientation: [pi, 0, 0]
    ])

    # Action space bounds (generous for demo collection)
    # _ABS_POSE_LIMIT_LOW = np.array([
    #     0.57, -0.093, 0.14,  # XYZ lower bounds
    #     np.pi - ROTATION_MARGIN, -ROTATION_MARGIN, -ROTATION_MARGIN
    # ])

    # _ABS_POSE_LIMIT_HIGH = np.array([
    #     0.585, -0.087, 0.19,  # XYZ upper bounds
    #     np.pi + ROTATION_MARGIN, ROTATION_MARGIN, ROTATION_MARGIN
    # ])

    # _ABS_POSE_LIMIT_LOW = np.array([
    #     0.555, -0.093, 0.14,  # XYZ lower bounds
    #     np.pi - ROTATION_MARGIN, -ROTATION_MARGIN, -ROTATION_MARGIN
    # ])
    # _ABS_POSE_LIMIT_HIGH = np.array([
    #     0.58, -0.08, 0.19,  # XYZ upper bounds
    #     np.pi + ROTATION_MARGIN, ROTATION_MARGIN, ROTATION_MARGIN
    # ])
    # 六边形
    _ABS_POSE_LIMIT_LOW = np.array([
        0.55, -0.093, 0.14,  # XYZ lower bounds
        np.pi - ROTATION_MARGIN, -ROTATION_MARGIN, -ROTATION_MARGIN
    ])
    _ABS_POSE_LIMIT_HIGH = np.array([
        0.585, -0.08, 0.19,  # XYZ upper bounds
        np.pi + ROTATION_MARGIN, ROTATION_MARGIN, ROTATION_MARGIN
    ])
    _POSE_CENTER = _RESET_POSE[:3].copy()

else:
    # ============================================
    # Auto-computed Pose Configuration (from demo data)
    # ============================================
    from utils.pose_limit_calculator import get_pose_limits

    DEMO_DATA_PATH = "/home/pi-zero/Documents/openpi/third_party/real_franka/data/peg_in_hole/peg_in_hole_square_III_1-13__mxp"

    _pose_limits = get_pose_limits(
        data_path=DEMO_DATA_PATH,
        peg_length=PEG_LENGTH,
        x_margin=X_MARGIN,
        y_margin=Y_MARGIN,
        rotation_margin=ROTATION_MARGIN,
    )

    _ABS_POSE_LIMIT_LOW = _pose_limits["low"]
    _ABS_POSE_LIMIT_HIGH = _pose_limits["high"]
    _POSE_CENTER = _pose_limits["center"]

    _TARGET_POSE = np.array([
        _POSE_CENTER[0],
        _POSE_CENTER[1],
        _ABS_POSE_LIMIT_LOW[2],
        np.pi, 0, 0
    ])

    _RESET_POSE = np.array([
        _POSE_CENTER[0],
        _POSE_CENTER[1],
        _ABS_POSE_LIMIT_HIGH[2],
        np.pi, 0, 0
    ])


# ============================================
# Camera Selection Configuration
# Uncomment the cameras you want to use for policy input
# ============================================
SELECTED_CAMERAS = [
    # "wrist_1",    # Wrist camera 1 (external view)
    "wrist_2",    # Wrist camera 2 (internal view)
    "side",       # Global side camera
    "top",        # Global top camera (will be cropped as top_crop)
]


class EnvConfig(DefaultEnvConfig):
    """
    Peg-in-Hole Square III Environment Configuration

    Optimized for precision insertion stage:
    - Uses cropped global camera (object-centric view)
    - Optional multi-view cameras
    - Small action scaling for fine control
    """

    SERVER_URL: str = "http://172.16.0.1:5000/"

    # ============================================
    # Camera Configuration (All available cameras)
    # ============================================
    _ALL_CAMERAS = {
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

    # Build REALSENSE_CAMERAS based on SELECTED_CAMERAS
    REALSENSE_CAMERAS = {k: v for k, v in _ALL_CAMERAS.items() if k in SELECTED_CAMERAS}

    # Image crop configuration
    IMAGE_CROP = {
        "wrist_1": lambda img: img,  # No crop for wrist
        "wrist_2": lambda img: img,  # No crop for wrist
        "side": lambda img: img,  # No crop
        "top": lambda img: img[214:400, 636:799],  # Object-centric crop
    }

    # ============================================
    # Key Position Configuration
    # TARGET_POSE: Goal pose for task (used for distance reward)
    # RESET_POSE: Reset pose at the start of each episode
    # ============================================
    TARGET_POSE = _TARGET_POSE
    RESET_POSE = _RESET_POSE

    # ============================================
    # Action Scaling
    # ============================================
    # [xy translation, rotation, gripper]
    # Rotation scaling set to 0 to keep orientation fixed at [pi, 0, 0]
    ACTION_SCALE = np.array([0.05, 0.0, 1])  # Increased translation, disabled rotation

    # ============================================
    # Randomization Configuration
    # For demo recording (MANUAL_POSE_CONFIG=True): disable random reset
    # For training (MANUAL_POSE_CONFIG=False): enable random reset
    # ============================================
    RANDOM_RESET = True  # Disable during demo recording

    # XY sampling range: use computed bounds rectangle
    RESET_X_RANGE = (_ABS_POSE_LIMIT_LOW[0], _ABS_POSE_LIMIT_HIGH[0])
    RESET_Y_RANGE = (_ABS_POSE_LIMIT_LOW[1], _ABS_POSE_LIMIT_HIGH[1])

    # Z sampling range: from ABS_POSE_LIMIT_HIGH[2] to +0.015m
    RESET_Z_RANGE = (
        _ABS_POSE_LIMIT_HIGH[2],          # Lower: top of action space
        _ABS_POSE_LIMIT_HIGH[2] + 0.015,  # Upper: 1.5cm above
    )

    RANDOM_RZ_RANGE = 0.0   # No rotation randomization
    DISPLAY_IMAGE = True

    # ============================================
    # Safety Bounds (Auto-computed from demo data)
    # X/Y: KDE mode +/- XY_MARGIN
    # Z: min(endpoint_z) to min(endpoint_z) + PEG_LENGTH
    # ============================================
    ABS_POSE_LIMIT_LOW = _ABS_POSE_LIMIT_LOW
    ABS_POSE_LIMIT_HIGH = _ABS_POSE_LIMIT_HIGH

    # ============================================
    # Compliance Control Parameters
    # ============================================
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "rotational_Ki": 0,
        "translational_clip_x": 0.012,
        "translational_clip_y": 0.012,
        "translational_clip_z": 0.010,
        "translational_clip_neg_x": 0.012,
        "translational_clip_neg_y": 0.012,
        "translational_clip_neg_z": 0.008,
        "rotational_clip_x": 0.030,
        "rotational_clip_y": 0.030,
        "rotational_clip_z": 0.025,
        "rotational_clip_neg_x": 0.030,
        "rotational_clip_neg_y": 0.030,
        "rotational_clip_neg_z": 0.025,
    }

    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.03,
        "rotational_clip_y": 0.03,
        "rotational_clip_z": 0.03,
        "rotational_clip_neg_x": 0.03,
        "rotational_clip_neg_y": 0.03,
        "rotational_clip_neg_z": 0.03,
        "rotational_Ki": 0.0,
    }

    MAX_EPISODE_LENGTH = 200  # ~20 seconds @ 10Hz


class TrainConfig(DefaultTrainingConfig):
    """
    Peg-in-Hole Square III Training Configuration

    Key features:
    1. Configurable image_keys for camera selection
    2. No classifier (manual keyboard judgement)
    3. Fixed gripper mode
    """

    # ============================================
    # Policy Image Keys - Based on SELECTED_CAMERAS
    # ============================================
    image_keys = SELECTED_CAMERAS.copy()

    # No automatic reward classifier
    classifier_keys = []

    # Proprioception features - Full state observation (19D)
    proprio_keys = [
        "tcp_pose",      # End-effector pose (6D: xyz + euler)
        "tcp_vel",       # End-effector velocity (6D)
        "tcp_force",     # Robot built-in force sensor (3D)
        "tcp_torque",    # Robot built-in torque sensor (3D)
        "gripper_pose",  # Gripper position (1D)
    ]

    # Training hyperparameters
    checkpoint_period = 1000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 2000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"  # Gripper always closed
    batch_size = 256
    max_traj_length = 200

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        """
        Build the Peg-in-Hole Square III environment

        Wrapper order:
        1. PegInHoleSquareIIIEnv (base environment)
        2. SpacemouseIntervention (human intervention)
        3. FixedPoseActionWrapper (lock rotation and gripper)
        4. RelativeFrame (relative coordinate system)
        5. Quat2EulerWrapper (quaternion to euler)
        6. SERLObsWrapper (standardize observation)
        7. ChunkingWrapper (observation history)
        8. KeyboardRewardWrapper (manual success judgement)
        """
        env_config = EnvConfig()

        # 1. Base environment
        env = PegInHoleSquareIIIEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=env_config
        )

        # 2. Human intervention (SpaceMouse)
        if not fake_env:
            env = SpacemouseIntervention(env)

        # 3. Fixed pose action (zero rotation, closed gripper)
        env = FixedPoseActionWrapper(env)

        # 4. Relative coordinate system
        env = RelativeFrame(env)

        # 5. Quaternion to Euler
        env = Quat2EulerWrapper(env)

        # 6. Standardize observation
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)

        # 7. Observation history
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

        # 8. Manual keyboard success judgement (replaces classifier)
        if classifier:
            env = KeyboardRewardWrapper(env, auto_query=True)

        return env


# ============================================
# Print computed values when module is imported (for verification)
# ============================================
def print_computed_config():
    """Print the auto-computed configuration values for verification."""
    print("\n" + "=" * 60)
    print("Peg-in-Hole Square III - Auto-computed Configuration")
    print("=" * 60)
    print(f"\nDemo data: {DEMO_DATA_PATH}")
    print(f"Peg length: {PEG_LENGTH*100:.1f}cm")
    print(f"X margin: -{X_MARGIN[0]*1000:.1f}mm / +{X_MARGIN[1]*1000:.1f}mm")
    print(f"Y margin: -{Y_MARGIN[0]*1000:.1f}mm / +{Y_MARGIN[1]*1000:.1f}mm")
    print(f"\nSelected cameras: {SELECTED_CAMERAS}")
    print(f"\nComputed bounds:")
    print(f"  ABS_POSE_LIMIT_LOW  = [{_ABS_POSE_LIMIT_LOW[0]:.4f}, {_ABS_POSE_LIMIT_LOW[1]:.4f}, {_ABS_POSE_LIMIT_LOW[2]:.4f}, ...]")
    print(f"  ABS_POSE_LIMIT_HIGH = [{_ABS_POSE_LIMIT_HIGH[0]:.4f}, {_ABS_POSE_LIMIT_HIGH[1]:.4f}, {_ABS_POSE_LIMIT_HIGH[2]:.4f}, ...]")
    print(f"\nComputed poses:")
    print(f"  TARGET_POSE = [{_TARGET_POSE[0]:.4f}, {_TARGET_POSE[1]:.4f}, {_TARGET_POSE[2]:.4f}, pi, 0, 0]")
    print(f"  RESET_POSE  = [{_RESET_POSE[0]:.4f}, {_RESET_POSE[1]:.4f}, {_RESET_POSE[2]:.4f}, pi, 0, 0]")
    print(f"\nReset sampling ranges:")
    print(f"  X: [{EnvConfig.RESET_X_RANGE[0]:.4f}, {EnvConfig.RESET_X_RANGE[1]:.4f}]")
    print(f"  Y: [{EnvConfig.RESET_Y_RANGE[0]:.4f}, {EnvConfig.RESET_Y_RANGE[1]:.4f}]")
    print(f"  Z: [{EnvConfig.RESET_Z_RANGE[0]:.4f}, {EnvConfig.RESET_Z_RANGE[1]:.4f}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_computed_config()
