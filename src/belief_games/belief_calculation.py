from belief_games import TwoVehicleSimpleParams
from decimal import Decimal as D

def belief_calc_simple1(p0: TwoVehicleSimpleParams):
    belief = {}
    p = D(1)-(D(1)/p0.side)*p0.first_progress
    q = D(1)-(D(1)/p0.side)*p0.second_progress

    belief["Game1"] = p
    belief["Game2"] = (D(1)-p)*q
    belief["Game3"] = (D(1)-p)*(D(1)-q)*D(0.5)
    belief["Game4"] = (D(1)-p)*(D(1)-q)*D(0.5)

    return belief

def belief_calc_simple2(p0: TwoVehicleSimpleParams):
    belief = {}
    p = D(1)-(D(1)/p0.side)*p0.second_progress
    q = D(1)-(D(1)/p0.side)*p0.first_progress

    belief["Game1"] = p
    belief["Game2"] = (D(1)-p)*q
    belief["Game3"] = (D(1)-p)*(D(1)-q)*D(0.5)
    belief["Game4"] = (D(1)-p)*(D(1)-q)*D(0.5)

    return belief
