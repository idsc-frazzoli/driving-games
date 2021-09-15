from typing import Sequence, Tuple, Optional

from dg_commons import DgSampledSequence
from games import X
from sim import Color

DrawableTrajectoryType = Sequence[Tuple[DgSampledSequence[X], Optional[Color]]]
"""The interface for the supported visualisation of trajectories. 
Each trajectory shall come paired with a color, setting to None will pick the agent's color """
