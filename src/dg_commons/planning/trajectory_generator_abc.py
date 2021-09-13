from abc import abstractmethod, ABC
from typing import Callable, Set, Optional

from dg_commons import Timestamp
from dg_commons.planning.trajectory import Trajectory
from sim.models.vehicle import VehicleState, VehicleCommands
from sim.models.vehicle_utils import VehicleParameters


class TrajGenerator(ABC):

    def __init__(self,
                 vehicle_dynamics: Callable[[VehicleState, VehicleCommands, Timestamp], VehicleState],
                 vehicle_params: VehicleParameters):
        self.vehicle_dynamics = vehicle_dynamics
        self.vehicle_params = vehicle_params

    @abstractmethod
    def generate(self, x0: Optional[VehicleState]) -> Set[Trajectory]:
        """Passing the current initial state is optional"""
        pass
