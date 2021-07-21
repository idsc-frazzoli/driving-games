import os

__all__ = ["config_dir", "pref_dir"]

config_dir = os.path.dirname(__file__)
pref_dir = os.path.join(config_dir, "player_pref")
