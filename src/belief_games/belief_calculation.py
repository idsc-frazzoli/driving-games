from belief_games import TwoVehicleSimpleParams
from decimal import Decimal as D

def belief_calc_simple1(p0: TwoVehicleSimpleParams):
    # This function caluclates the probabilities of playing the four games.

    # Initialize
    belief = {}

    # use positions of the two cars to calculate p,q. The more distance two the intersection, the more likely it is that somebody else is present.
    p = D(1)-(D(1)/p0.side)*p0.first_progress
    q = D(1)-(D(1)/p0.side)*p0.second_progress

    # Use p,q to determine the probabilities of each game. See my report for more information (c is 0.5 at the moment)
    belief["Game_Alone"] = p
    belief["Game_MS"] = (D(1)-p)*q
    belief["Game_SM"] = (D(1)-p)*(D(1)-q)*D(0.5)
    belief["Game_Normal"] = (D(1)-p)*(D(1)-q)*D(0.5)

    return belief

def belief_calc_simple2(p0: TwoVehicleSimpleParams):
    # Same as belief_calc_simple1, but for the second car.

    belief = {}
    p = D(1)-(D(1)/p0.side)*p0.second_progress
    q = D(1)-(D(1)/p0.side)*p0.first_progress

    belief["Game_Alone"] = p
    belief["Game_MS"] = (D(1)-p)*q
    belief["Game_SM"] = (D(1)-p)*(D(1)-q)*D(0.5)
    belief["Game_Normal"] = (D(1)-p)*(D(1)-q)*D(0.5)

    return belief
