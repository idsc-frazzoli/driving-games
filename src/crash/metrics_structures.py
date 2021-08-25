from dataclasses import dataclass
from typing import Mapping

from crash.metrics_malliaris import MalliarisRisk
from games import PlayerName


@dataclass(frozen=True, unsafe_hash=True)
class CollMetricsReport:
    malliaris: Mapping[PlayerName, MalliarisRisk]
    """ ... """
