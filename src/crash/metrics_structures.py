from dataclasses import dataclass
from typing import Mapping

from crash.metrics_utils import MalliarisRisk
from games import PlayerName


@dataclass(frozen=True, unsafe_hash=True)
class MetricsReport:
    malliaris: Mapping[PlayerName, MalliarisRisk]
    """ ... """
