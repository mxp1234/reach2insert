"""
SpaceMouse Intervention Module
Provides DAgger-style human intervention during policy execution.

This module allows human operators to override policy actions using a SpaceMouse
during inference. When the SpaceMouse is moved or buttons are pressed, the human
action takes precedence over the policy action.

Usage:
    intervention = SpaceMouseIntervention()

    # In the control loop:
    policy_action = policy.predict_action(obs)
    final_action, was_intervened, info = intervention.get_action(policy_action)
"""

import numpy as np
import time
from typing import Tuple, Dict, Any, Optional
from scripts.utils.spacemouse.spacemouse_expert import SpaceMouseExpert


class SpaceMouseIntervention:
    """
    SpaceMouse intervention handler for policy execution.

    This class wraps the SpaceMouseExpert and provides a simple interface
    for DAgger-style intervention during policy inference.

    Args:
        spacemouse_scale: Scale factor for spacemouse position commands (default: 0.05)
        policy_scale: Scale factor for policy position commands (default: 0.015)
        rotation_scale: Scale factor for rotation commands (default: 1.0)
        gripper_enabled: Whether to enable gripper control via buttons (default: True)
        intervention_threshold: Threshold for detecting spacemouse movement (default: 0.001)
        action_dim: Total action dimension including gripper (default: 7)
    """

    def __init__(
        self,
        spacemouse_scale: float = 0.05,
        policy_scale: float = 0.015,
        rotation_scale: float = 1.0,
        gripper_enabled: bool = True,
        intervention_threshold: float = 0.001,
        action_dim: int = 7,
    ):
        self.spacemouse_scale = spacemouse_scale
        self.policy_scale = policy_scale
        self.rotation_scale = rotation_scale
        self.gripper_enabled = gripper_enabled
        self.intervention_threshold = intervention_threshold
        self.action_dim = action_dim

        # Initialize SpaceMouse
        print("Initializing SpaceMouse...")
        try:
            self.expert = SpaceMouseExpert()
            self.connected = True
            print("  SpaceMouse connected successfully")
        except Exception as e:
            print(f"  Warning: Failed to connect SpaceMouse: {e}")
            self.expert = None
            self.connected = False

        # Button states
        self.left_button = False
        self.right_button = False

        # Statistics
        self.total_steps = 0
        self.intervention_steps = 0
        self.intervention_episodes = 0
        self._last_intervene_time = 0
        self._intervention_active = False

    def get_action(
        self,
        policy_action: np.ndarray,
        scale_policy: bool = True,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Get the action to execute, either from policy or human intervention.

        Args:
            policy_action: The action predicted by the policy [dx, dy, dz, drx, dry, drz, gripper]
            scale_policy: Whether to apply policy_scale to policy actions (default: True)

        Returns:
            action: The final action to execute
            intervened: True if human intervened
            info: Dictionary containing:
                - 'left_button': Left button state
                - 'right_button': Right button state
                - 'intervene_action': The intervention action (if intervened)
                - 'raw_spacemouse': Raw spacemouse values
        """
        self.total_steps += 1
        info = {
            'left_button': False,
            'right_button': False,
            'raw_spacemouse': np.zeros(6),
        }

        # If spacemouse not connected, return policy action
        if not self.connected or self.expert is None:
            action = policy_action.copy()
            if scale_policy:
                action[:3] = action[:3] * self.policy_scale
            return action, False, info

        # Get spacemouse input
        expert_action, buttons = self.expert.get_action()
        expert_action = expert_action[:6]  # Only take first 6 DoF

        info['raw_spacemouse'] = expert_action.copy()

        # Parse button states
        if len(buttons) >= 2:
            self.left_button = bool(buttons[0])
            self.right_button = bool(buttons[1])
        info['left_button'] = self.left_button
        info['right_button'] = self.right_button

        # Check if spacemouse is being used
        spacemouse_active = np.linalg.norm(expert_action) > self.intervention_threshold
        button_active = self.left_button or self.right_button
        intervened = spacemouse_active or button_active

        if intervened:
            # Build intervention action
            action = np.zeros(self.action_dim, dtype=np.float32)

            # Position: scale spacemouse input
            action[:3] = expert_action[:3] * self.spacemouse_scale

            # Rotation: scale spacemouse input
            action[3:6] = expert_action[3:6] * self.spacemouse_scale * self.rotation_scale

            # Gripper control via buttons
            # Only change gripper when button is pressed, otherwise keep policy's gripper value
            if self.gripper_enabled and self.action_dim >= 7:
                if self.left_button:
                    # Close gripper (negative value)
                    action[6] = np.random.uniform(-1, -0.9)
                elif self.right_button:
                    # Open gripper (positive value)
                    action[6] = np.random.uniform(0.9, 1)
                else:
                    # No button pressed - keep policy's gripper value
                    action[6] = policy_action[6] if len(policy_action) > 6 else 0.0

            info['intervene_action'] = action.copy()

            # Update statistics
            self.intervention_steps += 1
            if not self._intervention_active:
                self.intervention_episodes += 1
                self._intervention_active = True
            self._last_intervene_time = time.time()

            return action, True, info
        else:
            # Use policy action
            action = policy_action.copy()
            if scale_policy:
                action[:3] = action[:3] * self.policy_scale

            # Reset intervention tracking if no recent intervention
            if time.time() - self._last_intervene_time > 0.5:
                self._intervention_active = False

            return action, False, info

    def get_intervention_stats(self) -> Dict[str, Any]:
        """Get intervention statistics."""
        return {
            'total_steps': self.total_steps,
            'intervention_steps': self.intervention_steps,
            'intervention_episodes': self.intervention_episodes,
            'intervention_rate': self.intervention_steps / max(1, self.total_steps),
        }

    def reset_stats(self):
        """Reset intervention statistics."""
        self.total_steps = 0
        self.intervention_steps = 0
        self.intervention_episodes = 0
        self._intervention_active = False

    def close(self):
        """Close the SpaceMouse connection."""
        if self.expert is not None:
            self.expert.close()
            self.connected = False


class SpaceMouseInterventionWithInertia(SpaceMouseIntervention):
    """
    SpaceMouse intervention with inertia - continues using human action for
    a short time after spacemouse becomes inactive.

    This helps with smoother transitions and prevents jittery behavior when
    the human releases the spacemouse.

    Args:
        inertia_duration: How long to continue using human action after release (seconds)
        **kwargs: Additional arguments passed to SpaceMouseIntervention
    """

    def __init__(self, inertia_duration: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.inertia_duration = inertia_duration
        self._last_intervene_action = None

    def get_action(
        self,
        policy_action: np.ndarray,
        scale_policy: bool = True,
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Get action with inertia support."""

        # First check raw spacemouse state
        if self.connected and self.expert is not None:
            expert_action, buttons = self.expert.get_action()
            expert_action = expert_action[:6]

            spacemouse_active = np.linalg.norm(expert_action) > self.intervention_threshold
            button_active = bool(buttons[0]) or bool(buttons[1]) if len(buttons) >= 2 else False

            if spacemouse_active or button_active:
                # Active intervention
                action, intervened, info = super().get_action(policy_action, scale_policy)
                self._last_intervene_action = action.copy()
                self._last_intervene_time = time.time()
                return action, intervened, info

        # Check if within inertia window
        if (self._last_intervene_action is not None and
            time.time() - self._last_intervene_time < self.inertia_duration):
            # Still within inertia window, use last intervention action
            info = {
                'left_button': self.left_button,
                'right_button': self.right_button,
                'raw_spacemouse': np.zeros(6),
                'intervene_action': self._last_intervene_action.copy(),
                'inertia': True,
            }
            self.intervention_steps += 1
            return self._last_intervene_action.copy(), True, info

        # No intervention, use policy
        return super().get_action(policy_action, scale_policy)
