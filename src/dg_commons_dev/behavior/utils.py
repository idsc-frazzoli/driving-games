from dataclasses import dataclass
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim import PlayerObservations
from typing import MutableMapping, Dict, Optional


@dataclass
class SituationObservations:
    my_name: PlayerName

    agents: Optional[MutableMapping[PlayerName, PlayerObservations]] = None

    rel_poses: Optional[Dict[PlayerName, SE2Transform]] = None
