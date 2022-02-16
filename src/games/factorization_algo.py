from typing import Mapping, Tuple, FrozenSet as FSet, Dict

from cytoolz import itemmap

from dg_commons import PlayerName, X, fd, U, Y, RP, RJ
from games import JointState, SR
from games.factorization import find_dependencies
from games.solve.solution_structures import UsedResources, SolvedGameNode, FactAlgo
from possibilities import PossibilityMonad

__all__ = ["FactAlgoReachableRes", "FactAlgoNoFact"]


class FactAlgoReachableRes(FactAlgo):
    """Factorize according to the reachable resources"""

    def factorize(
        self,
        js0: JointState,
        known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]],
        ps: PossibilityMonad,
    ) -> Mapping[PlayerName, JointState]:
        fact_states: Dict[PlayerName, JointState] = {}

        if len(js0) > 1:

            def get_reachable_res(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
                pname, state = items
                alone_js = fd({pname: state})
                return pname, known[pname][alone_js].reachable_res

            resources_used = itemmap(get_reachable_res, js0)
            deps: Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]
            deps = find_dependencies(ps, resources_used, self.f_resource_intersection)

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
    """This is the vanilla implementation that does *not* factorize"""

    def factorize(
        self,
        js0: JointState,
        known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]],
        ps: PossibilityMonad,
    ) -> Mapping[PlayerName, JointState]:
        js1: Dict[PlayerName, JointState]
        js1 = {k_: js0 for k_ in js0}
        return fd(js1)
