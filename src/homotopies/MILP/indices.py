from enum import IntEnum, unique


@unique
class IdxInput(IntEnum):
    Sigma_L = 0
    """binary variable indicating whether vehicle is to the left of the obstacle"""
    Sigma_R = 1
    """binary variable indicating whether vehicle is to the right of the obstacle"""
    Sigma_B = 2
    """binary variable indicating whether vehicle is below the obstacle"""
    Sigma_A = 3
    """binary variable indicating whether vehicle is to the left of the obstacle"""
    Slack_L = 4
    """slack variable for sigma_L"""
    Slack_R = 5
    """slack variable for sigma_R"""
    Slack_B = 6
    """slack variable for sigma_B"""
    Slack_A = 7
    """slack variable for sigma_A"""
    ddS = 8
    """lateral acc"""


@unique
class IdxState(IntEnum):
    S = 9
    """x-position in world frame"""
    dS = 10
