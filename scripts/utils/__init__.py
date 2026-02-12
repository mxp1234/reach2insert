"""Utility modules for DP-SERL training."""

from .tactile_utils import TactileBaselineManager, TACTILE_AVAILABLE
from .robot_utils import (
    get_robot_state,
    send_action,
    clear_robot_error,
    open_gripper,
    close_gripper,
    update_compliance_param,
    check_in_serl_space,
    precise_wait,
    reset_robot_to_position,
)
from .camera_utils import RealSenseCamera, MultiCameraSystem, process_image_dp, process_image_serl
from .dp_inference import DPInference, ActionQueue, GripperSmoother

# TactileSensor (optional - requires pyserial)
try:
    from .tactile_sensor import TactileSensor, create_tactile_sensor
except ImportError:
    TactileSensor = None
    create_tactile_sensor = None

# SpaceMouse (optional - requires easyhid)
try:
    from .spacemouse import SpaceMouseExpert, SpaceMouseIntervention, SpaceMouseInterventionWithInertia
    SPACEMOUSE_AVAILABLE = True
except ImportError:
    SpaceMouseExpert = None
    SpaceMouseIntervention = None
    SpaceMouseInterventionWithInertia = None
    SPACEMOUSE_AVAILABLE = False

__all__ = [
    # Tactile
    "TactileBaselineManager",
    "TactileSensor",
    "create_tactile_sensor",
    "TACTILE_AVAILABLE",
    # SpaceMouse
    "SpaceMouseExpert",
    "SpaceMouseIntervention",
    "SpaceMouseInterventionWithInertia",
    "SPACEMOUSE_AVAILABLE",
    # Robot
    "get_robot_state",
    "send_action",
    "clear_robot_error",
    "open_gripper",
    "close_gripper",
    "update_compliance_param",
    "check_in_serl_space",
    "precise_wait",
    "reset_robot_to_position",
    # Camera
    "RealSenseCamera",
    "MultiCameraSystem",
    "process_image_dp",
    "process_image_serl",
    # DP
    "DPInference",
    "ActionQueue",
    "GripperSmoother",
]
