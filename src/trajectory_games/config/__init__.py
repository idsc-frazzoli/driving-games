import os

__all__ = ["CONFIG_DIR", "PREF_DIR"]

CONFIG_DIR = os.path.dirname(__file__)
PREF_DIR = os.path.join(CONFIG_DIR, "player_pref")
