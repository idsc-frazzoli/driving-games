from dataclasses import dataclass
from typing import Mapping, List, Dict

from games import PlayerName
from world import Lane, NodeName, load_driving_game_map, get_lane_from_node_sequence
from .structures import VehicleGeometry

__all__ = ["TrajectoryWorld"]


@dataclass
class TrajectoryWorld:
    """ Object holding all info about the world """

    map_name: str
    """ Map name for the world """

    geo: Mapping[PlayerName, VehicleGeometry]
    """ Geometry of each player """

    nodes: Mapping[PlayerName, List[NodeName]]
    """ Reference node points for each player """

    lanes: Mapping[PlayerName, Lane] = None
    """ Reference lanes for each player """

    def __post_init__(self):
        assert self.nodes.keys() == self.geo.keys(), \
            f"Keys for geometry and nodes do not match - " \
            f"nodes = {self.nodes.keys()}, geo = {self.geo.keys()}"
        duckie_map = load_driving_game_map(self.map_name)
        lanes: Dict[PlayerName, Lane] = {}
        for player in self.get_players():
            lanes[player] = get_lane_from_node_sequence(m=duckie_map,
                                                        node_sequence=self.nodes[player])
        self.lanes = lanes

    def get_players(self) -> List[PlayerName]:
        return list(self.geo.keys())

    def get_lane(self, player: PlayerName) -> Lane:
        return self.lanes[player]

    def get_geometry(self, player: PlayerName) -> VehicleGeometry:
        return self.geo[player]
