"""
DP-SERL Curriculum Training Scripts

Modular implementation with:
- Demo data processing with tactile baseline extraction
- Grouped replay buffer with interval-based grouping (Mx, My)
- Offline/Online sampling with annealing
- BC + Critic pretraining
"""

from .config import TrainingConfig, CurriculumConfig
from .demo_processor import DemoProcessor

# Optional imports (require gymnasium)
try:
    from .grouped_buffer import GroupedReplayBuffer
    from .grouped_sampler import GroupedSampler
    from .pretrainer import Pretrainer
    _FULL_IMPORTS = True
except ImportError:
    GroupedReplayBuffer = None
    GroupedSampler = None
    Pretrainer = None
    _FULL_IMPORTS = False

__all__ = [
    "TrainingConfig",
    "CurriculumConfig",
    "DemoProcessor",
    "GroupedReplayBuffer",
    "GroupedSampler",
    "Pretrainer",
]
