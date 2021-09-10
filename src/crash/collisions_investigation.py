from functools import reduce
from itertools import permutations
from operator import attrgetter
from typing import List, Set, Tuple

import networkx as nx
from networkx import DiGraph

from crash import logger
from games import PlayerName
from sim import CollisionReport
from sim.collision_structures import combine_collision_reports


def _first_collision_for(player: PlayerName, coll_report: List[CollisionReport]):
    return min([report.at_time for report in coll_report if player in report.players])


def investigate_collision_report(coll_reports: List[CollisionReport]) \
        -> (List[CollisionReport], DiGraph):
    """We get a collision report for every step of the simulation in which a collision is detected.
    Yet an accident is *one* accident even if the two cars are in a collision state for multiple simulation steps.
    This function aims to compress the list of collision report to a list of "accidents"
    Assumptions:
        - For each episode two players can have an accidents with each other only once
    :param coll_reports: The original list of collision reports
    """

    players_involved: Set[PlayerName] = set()
    accidents: Set[Tuple[PlayerName]] = set()
    for report in coll_reports:
        players = tuple(report.players.keys())
        for p in players:
            players_involved.add(p)
        accidents.add(players)
    logger.info(f"From {len(coll_reports)} collisions "
                f"we detected {len(accidents)} accidents involving {len(players_involved)} players.")

    # We represent the chain of accidents as a direct graph
    Gcoll = DiGraph()
    for player in players_involved:
        ts_first_collision = min([report.at_time for report in coll_reports if player in report.players])
        Gcoll.add_node(player, ts_first_collision=ts_first_collision)

    for players in accidents:
        # we add an edge from p1->p2 iff p1["ts_first_collision"]<=p2["ts_first_collision"]
        for p1, p2 in permutations(players):
            if Gcoll.nodes[p1]["ts_first_collision"] <= Gcoll.nodes[p2]["ts_first_collision"]:
                Gcoll.add_edge(p1, p2)

    logger.info("Combining collisions reports into accidents' reports")
    list_accidents: List[CollisionReport] = []
    for involved_ps in accidents:
        report_involved_ps = [r for r in coll_reports if set(involved_ps) == set(r.players.keys())]
        accident_report = reduce(combine_collision_reports, report_involved_ps)
        list_accidents.append(accident_report)
    coll_reports = list_accidents

    # add info to node graph about who is at fault
    for player in Gcoll.nodes:
        # sort by time
        coll_reports.sort(key=attrgetter("at_time"))
        is_p_at_fault = next(r.players[player].at_fault for r in coll_reports if player in r.players)
        nx.set_node_attributes(Gcoll, {player: {"at_fault": is_p_at_fault}})

    return coll_reports, Gcoll
