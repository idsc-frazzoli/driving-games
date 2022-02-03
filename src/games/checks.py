from frozendict import frozendict
from zuper_commons.types import check_isinstance, ZValueError

from games import GameConstants, JointState
from games.game_def import PlayerOptions, JointPureActions, JointMixedActions
from possibilities import Poss


def check_joint_state(js: JointState, **kwargs):
    """Checks js is a :any:`JointState`."""
    if not GameConstants.checks:
        return

    check_isinstance(js, frozendict)
    for n, x in js.items():
        check_isinstance(n, str, **kwargs)
        if x is None:
            raise ZValueError(js=js, **kwargs)


def check_player_options(a: PlayerOptions, **kwargs):
    """Checks consistency of a PlayerOptions variable."""
    if not GameConstants.checks:
        return

    check_isinstance(a, frozendict, **kwargs)
    for k, v in a.items():
        check_isinstance(k, str)
        check_isinstance(v, frozenset)


def check_joint_pure_actions(a: JointPureActions, **kwargs):
    """Checks consistency of a JointPureActions variable."""
    if not GameConstants.checks:
        return

    check_isinstance(a, frozendict, **kwargs)
    if len(a) == 0:
        raise ZValueError("empty actions", a=a)
    for k, v in a.items():
        assert isinstance(k, str), k
        if isinstance(v, Poss):
            msg = "I thought this would be pure actions, found Poss inside"
            raise ZValueError(msg, k=k, v=v, **kwargs)


def check_joint_mixed_actions(a: JointMixedActions, **kwargs):
    """Checks consistency of a JointMixedActions variable."""
    if not GameConstants.checks:
        return
    check_isinstance(a, frozendict, **kwargs)

    for k, v in a.items():
        check_isinstance(k, str)  # player name
        check_isinstance(v, Poss, **kwargs)
        for _ in v.support():
            if isinstance(_, Poss):
                raise ZValueError(_=_, **kwargs)
