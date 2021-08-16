from dataclasses import dataclass
from typing import Mapping

from games import PlayerName


@dataclass(frozen=True, unsafe_hash=True)
class MalliarisOneReportPlayer:
    p_fatality: float
    """ Probability of fatality """
    p_mais3: float
    """ Probability of MAIS 3+ injury """
    p_mais2: float
    """ Probability of MAIS 2+ injury """

    def __str__(self):
        return f"Prob. of fatality: {self.p_fatality*100:.2f}%\n" \
               f"Prob. of MAIS 3+ injury: {self.p_mais3*100:.2f}%\n" \
               f"Prob. of MAIS 2+ injury: {self.p_mais2*100:.2f}%\n"


@dataclass(frozen=True, unsafe_hash=True)
class MetricsReport:
    malliaris: Mapping[PlayerName, MalliarisOneReportPlayer]
    """ ... """
