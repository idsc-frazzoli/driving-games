from abc import abstractmethod, ABC
from typing import Mapping, Tuple, FrozenSet as FSet, Dict, Callable

from cytoolz import itemmap

from dg_commons import PlayerName, X, fd
from games import JointState, SR
from games.create_joint_game_tree_fact import IterationContextFact
from games.factorization import find_dependencies
from games.solve.solution_structures import UsedResources
from possibilities import PossibilityMonad


class FactAlgo(ABC):

    f_resource_intersection: Callable[[FSet[SR], FSet[SR]], bool]
    """Function to check for intersection of resources"""

    @staticmethod
    @abstractmethod
    def factorize(s0: JointState, ic: IterationContextFact, ps: PossibilityMonad) -> Mapping[PlayerName, JointState]:
        pass


class FactAlgoReachableRes(FactAlgo):
    @staticmethod
    def factorize(js0: JointState, ic: IterationContextFact, ps: PossibilityMonad) -> Mapping[PlayerName, JointState]:
        fact_states: Dict[PlayerName, JointState] = {}

        if len(js0) > 1:

            def get_reachable_res(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
                pname, state = items
                alone_js = fd({pname: state})
                return pname, ic.known[pname][alone_js].reachable_res

            resources_used = itemmap(get_reachable_res, js0)
            deps: Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]
            deps = find_dependencies(ps, resources_used, ic.f_resource_intersection)

            pset: FSet[PlayerName]
            for pset in deps[frozenset(js0)]:
                jsf: JointState = fd({p: js0[p] for p in pset})
                for p in pset:
                    fact_states[p] = jsf
            # logger.info(deps=deps, fact_states=fact_states)
        else:
            for p in js0:
                fact_states[p] = js0
        return fd(fact_states)


class FactAlgoNoFact(FactAlgo):
    @staticmethod
    def factorize(js0: JointState, ic: IterationContextFact, ps: PossibilityMonad) -> Mapping[PlayerName, JointState]:
        js1: Dict[PlayerName, JointState]
        js1 = {k_: js0 for k_ in js0}
        return fd(js1)
