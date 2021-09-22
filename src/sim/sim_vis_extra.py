from typing import Sequence, Tuple

from dg_commons import DgSampledSequence
from dg_commons import X
from sim import Color

DrawableTrajectoryType = Sequence[Tuple[DgSampledSequence[X], Color]]
"""The interface for the supported visualisation of trajectories. 
Each trajectory shall come paired with a color, setting to None will pick the agent's color """
