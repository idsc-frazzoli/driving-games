from dataclasses import replace
from typing import Mapping, Tuple, FrozenSet as FSet, Dict

from cytoolz import itemmap

from dg_commons import PlayerName, X, fd, U, Y, RP, RJ
from games import JointState, SR
from games.factorization import find_dependencies
from games.solve.solution_structures import UsedResources, SolvedGameNode, FactAlgo
from possibilities import PossibilityMonad

__all__ = ["FactAlgoReachableRes", "FactAlgoNoFact", "FactAlgoOptimalRes"]


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
                # breaks generality
                state = replace(state, has_collided=False) if state.has_collided else state
                alone_js = fd({pname: state})
                # if alone_js not in known[pname]:
                #     raise ZValueError(
                #         f"{alone_js} not in known",
                #         known=list(known[pname].keys()),
                #     )
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


class FactAlgoOptimalRes(FactAlgo):
    """Factorize according to the optimal resources"""

    def factorize(
        self,
        js0: JointState,
        known: Mapping[PlayerName, Mapping[JointState, SolvedGameNode[X, U, Y, RP, RJ, SR]]],
        ps: PossibilityMonad,
    ) -> Mapping[PlayerName, JointState]:
        fact_states: Dict[PlayerName, JointState] = {}

        if len(js0) > 1:
            js0_minus_collision: Dict[PlayerName, X] = dict(js0)
            # so that they can be found in the known
            for p, x in js0_minus_collision.items():
                if x.has_collided:  # breaks generality
                    js0_minus_collision[p] = replace(x, has_collided=False)

            def get_optimal_res(items: Tuple[PlayerName, X]) -> Tuple[PlayerName, UsedResources]:
                pname, state = items
                alone_js = fd({pname: state})
                return pname, known[pname][alone_js].optimal_res

            opt_resources_used = itemmap(get_optimal_res, js0_minus_collision)
            deps: Mapping[FSet[PlayerName], FSet[FSet[PlayerName]]]
            deps = find_dependencies(ps, opt_resources_used, self.f_resource_intersection)
            candidate_fact = deps[frozenset(js0)]
            current_n_pset = 1
            while len(candidate_fact) > current_n_pset:
                current_n_pset = len(candidate_fact)

                def get_opt_or_reachable_res(pset: FSet[PlayerName]) -> Mapping[PlayerName, UsedResources]:
                    used_res: Dict[PlayerName, UsedResources] = {}
                    for pname in pset:
                        alone_js = fd({pname: js0_minus_collision[pname]})
                        pname_res = (
                            known[pname][alone_js].optimal_res
                            if len(pset) == 1
                            else known[pname][alone_js].reachable_res
                        )
                        used_res.update({pname: pname_res})

                    return fd(used_res)

                resources_used = {}
                for pset in candidate_fact:
                    resources_used.update(get_opt_or_reachable_res(pset))
                deps = find_dependencies(ps, fd(resources_used), self.f_resource_intersection)
                candidate_fact = deps[frozenset(js0)]

            pset: FSet[PlayerName]  # a partition of players
            for pset in candidate_fact:
                jsf: JointState = fd({p: js0[p] for p in pset})
                for p in pset:
                    fact_states[p] = jsf
            # logger.info(deps=deps, fact_states=fact_states)
        else:
            for p in js0:
                fact_states[p] = js0
        return fd(fact_states)
