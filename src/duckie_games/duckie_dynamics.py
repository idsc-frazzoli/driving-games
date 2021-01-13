from typing import Mapping, FrozenSet
import itertools
from decimal import Decimal as D
from functools import lru_cache
from frozendict import frozendict
from zuper_commons.types import ZValueError, ZException

from driving_games.structures import (
    SE2_disc,
    LightsValue,
    VehicleState,
    VehicleActions,
    Lights,
    VehicleGeometry,
)

from possibilities import Poss, PossibilityMonad

from driving_games.vehicle_dynamics import VehicleDynamics
from driving_games.rectangle import Rectangle

from .structures import DuckieState, DuckieActions, DuckieGeometry



class DuckieDynamics(VehicleDynamics):
    def __init__(
            self,
            max_speed: D,
            min_speed: D,
            available_accels: FrozenSet[D],
            max_wait: D,
            ref: SE2_disc,
            max_path: D,
            lights_commands: FrozenSet[Lights],
            shared_resources_ds: D,
            vg: DuckieGeometry,
            poss_monad: PossibilityMonad,
    ):
        VehicleDynamics.__init__(
            self=self,
            max_speed=max_speed,
            min_speed=min_speed,
            available_accels=available_accels,
            max_wait=max_wait,
            ref=ref,
            max_path=max_path,
            lights_commands=lights_commands,
            shared_resources_ds=shared_resources_ds,
            vg=vg,
            poss_monad=poss_monad,
        )
