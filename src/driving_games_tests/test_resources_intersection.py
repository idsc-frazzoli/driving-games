from decimal import Decimal as D

from dg_commons.sim.models.vehicle_ligths import NO_LIGHTS
from driving_games import VehicleTrackState
from driving_games.zoo_games import games_zoo, P1, P2


def test_1():
    game_sc = games_zoo["4way_int_3p_sets"]

    p1_dyn = game_sc.game.players[P1].dynamics
    p2_dyn = game_sc.game.players[P2].dynamics
    x1: VehicleTrackState = VehicleTrackState(x=D(140.0), v=D(2.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    p1_res = p1_dyn.get_shared_resources(x1)
    x2: VehicleTrackState = VehicleTrackState(x=D(185.0), v=D(2.0), wait=D(0), light=NO_LIGHTS, has_collided=False)
    p2_res = p2_dyn.get_shared_resources(x2)
    print(p1_res)
    print(p2_res)
    print(p1_res & p2_res)
