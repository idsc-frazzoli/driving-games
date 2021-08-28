from dataclasses import dataclass
from typing import Mapping

from crash.metrics_malliaris import MalliarisRisk
from crash.metrics_pedestrian import PedestrianRisk
from games import PlayerName


@dataclass(frozen=True, unsafe_hash=True)
class CollMetricsReport:
    malliaris: Mapping[PlayerName, MalliarisRisk]
    """ ... """
    pedestrian: Mapping[PlayerName, PedestrianRisk]
    """ ... """