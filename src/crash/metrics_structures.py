from dataclasses import dataclass
from typing import Mapping

from crash.metrics_malliaris import MalliarisRisk
from crash.metrics_nieliyang import NieLiYangRisk
from dg_commons import PlayerName


@dataclass(frozen=True, unsafe_hash=True)
class CollMetricsReport:
    malliaris: Mapping[PlayerName, MalliarisRisk]
    """ Malliaris model fro car-to-car collisions """
    nieliyang: Mapping[PlayerName, NieLiYangRisk]
    """ NieLiYang model for car-to-pedestrian/cyclist collisions """
