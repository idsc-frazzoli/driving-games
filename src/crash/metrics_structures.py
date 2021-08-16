from dataclasses import dataclass
from typing import Mapping

from games import PlayerName
from sim import CollisionReportPlayer


@dataclass(frozen=True, unsafe_hash=True)
class MalliarisOneReportPlayer:
    p_fatality: float
    """ Probability of fatality """
    p_mais3: float
    """ Probability of MAIS 3+ injury """
    p_mais2: float
    """ Probability of MAIS 2+ injury """


@dataclass(frozen=True, unsafe_hash=True)
class MetricsReport:
    players: Mapping[PlayerName, MalliarisOneReportPlayer]
    """ ... """
