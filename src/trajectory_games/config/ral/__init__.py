import os

__all__ = ["config_dir_ral", "pref_dir_ral"]

config_dir_ral = os.path.dirname(__file__)
pref_dir_ral = os.path.join(config_dir_ral, "player_pref")
