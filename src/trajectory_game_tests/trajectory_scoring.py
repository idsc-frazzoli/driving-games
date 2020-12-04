from typing import List, Tuple, Dict, Mapping
from time import perf_counter
from decimal import Decimal as D

from preferences import Preference, remove_dominated, StrictProductPreferenceDict, SmallerPreferred
from trajectory_game import World, SplineTransitionPath, Trajectory, AllTrajectories, RuleEvaluationResult


def create_highway_world() -> World:
    s: List[float] = [float(_) for _ in range(10)]
    x: List[float] = [0.0 for _ in range(10)]
    y: List[float] = s
    ref = SplineTransitionPath[float](s=s, x=x, y=y, order=3)
    left: List[Tuple[float, float]] = [(-5.0, float(_)) for _ in range(10)]
    right: List[Tuple[float, float]] = [(5.0, float(_)) for _ in range(10)]
    world = World(ref_path=ref, left_xy=left, right_xy=right)
    return world


def score_trajectories(all_traj: AllTrajectories) -> Dict[Trajectory, Dict[str, RuleEvaluationResult]]:
    tic = perf_counter()
    world = create_highway_world()
    toc = perf_counter() - tic
    print("World generation time = {} s".format(toc))

    tic = perf_counter()
    result = all_traj.evaluate_trajectories(world=world)
    toc = perf_counter() - tic
    print("Metric evaluation time = {} s".format(toc))
    return result


def remove_dominated_trajectories(
    result: Dict[Trajectory, Dict[str, RuleEvaluationResult]]
) -> Dict[Trajectory, Dict[str, RuleEvaluationResult]]:
    pref_list: List[str] = []
    pref_list_temp: List[str] = []
    all_prefs: Dict[Trajectory, Mapping[str, Preference[D]]] = {}
    tic = perf_counter()
    for traj, _ in result.items():
        scores: Dict[str, D] = {}
        for __, rer in _.items():
            for ___, metric in rer.metrics.items():
                key: str = metric.title
                if not pref_list:
                    pref_list_temp.append(key)
                scores[key] = D(metric.total)
        if not pref_list:
            pref_list = pref_list_temp
        all_prefs[traj] = scores

    toc = perf_counter() - tic
    print("pref dict generation time = {} s".format(toc))
    pref_dict: Dict[str, Preference[D]] = {p: SmallerPreferred() for p in pref_list}
    pref = StrictProductPreferenceDict(prefs=pref_dict)
    tic = perf_counter()
    nondom_traj = remove_dominated(orig=all_prefs, pref=pref)
    toc = perf_counter() - tic
    print("Remove dominated time = {} s".format(toc))

    ret: Dict[Trajectory, Dict[str, RuleEvaluationResult]] = {}
    for key, _ in nondom_traj.items():
        ret[key] = result[key]

    return ret
