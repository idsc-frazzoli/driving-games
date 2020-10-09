from typing import NewType

__all__ = ["PlayerBelief"]

PlayerBelief = NewType("PlayerBelief", float)
""" A belief for the player"""
# fixme I would expect this to be a distribution... not a float
