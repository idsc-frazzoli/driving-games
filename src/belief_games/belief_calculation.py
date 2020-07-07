from belief_games import TwoVehicleSimpleParams


def belief_calc_simple1(p0: TwoVehicleSimpleParams):
    belief = {}
    p = 1-(1/p0.side)*p0.first_progress
    q = 1-(1/p0.side)*p0.second_progress

    belief["Game1"] = p
    belief["Game2"] = (1-p)*q
    belief["Game3"] = (1-p)*(1-q)*0.5
    belief["Game4"] = (1-p)*(1-q)*0.5

    return belief

def belief_calc_simple2(p0: TwoVehicleSimpleParams):
    belief = {}
    p = 1-(1/p0.side)*p0.second_progress
    q = 1-(1/p0.side)*p0.first_progress

    belief["Game1"] = p
    belief["Game2"] = (1-p)*q
    belief["Game3"] = (1-p)*(1-q)*0.5
    belief["Game4"] = (1-p)*(1-q)*0.5

    return belief
