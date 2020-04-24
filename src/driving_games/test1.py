from collections import defaultdict
from itertools import product

import networkx as nx
from networkx import convert_node_labels_to_integers, DiGraph
from networkx.drawing.nx_pydot import graphviz_layout

from reprep import Report
from zuper_commons.types import check_isinstance, ZException
from . import logger
from .driving_example import get_game1, VehicleState
from .game_def import Game, GamePlayer, PlayerName


def test1_a():
    game = get_game1()
    logger.info(game=game)

    r = Report()
    for player_name, player in game.players.items():
        r.add_child(report_player(player_name, player))
        break
    r.add_child(report_game(game))
    r.to_html('out/a.html')


def get_player_graph(player: GamePlayer) -> DiGraph:
    G = DiGraph()

    # all_states = player.dynamics.all_states()
    # logger.info(all_states=sorted(all_states))
    # for s in all_states:
    #     is_final = player.personal_reward_structure.is_personal_final_state(s)
    #     G.add_node(s, is_final=is_final)
    #
    # all_states = player.dynamics.all_states()
    # for s in all_states:
    #     is_final = player.personal_reward_structure.is_personal_final_state(s)
    #     assert s in G.nodes
    #     if not is_final:
    #         successors = player.dynamics.successors(s)
    #         for u, s2s in successors.items():
    #             for s2 in s2s:
    #                 assert s2 in all_states, (s, u, s2)
    #                 assert s2 in G.nodes, (s, u, s2)
    #                 G.add_edge(s, s2, u=u)

    for i in player.initial:
        i_final = player.personal_reward_structure.is_personal_final_state(i)
        if i_final:
            raise ZException(i_final=i_final)

        G.add_node(i, is_final=False)
    stack = list(player.initial)
    logger.info(stack=stack)
    i = 0
    while stack:
        print(i, len(stack), len(G.nodes))
        i += 1
        s1 = stack.pop(0)
        assert s1 in G.nodes

        # is_final =  player.personal_reward_structure.is_personal_final_state(s1)
        # G.add_node(s1, is_final=is_final)
        # # logger.info(s1=G.nodes[s1])

        successors = player.dynamics.successors(s1)
        for u, s2s in successors.items():
            for s2 in s2s:
                if s2 not in G.nodes:
                    is_final2 = player.personal_reward_structure.is_personal_final_state(s2)
                    G.add_node(s2, is_final=is_final2)
                    if not is_final2:
                        stack.append(s2)

                G.add_edge(s1, s2, u=u)
    return G


def report_player(player_name: PlayerName, player: GamePlayer):
    G = get_player_graph(player)
    G2 = convert_node_labels_to_integers(G)
    for (n1, n2, d) in G2.edges(data=True):
        d.clear()
    nx.write_graphml(G2, f'player-{player_name}.graphml')

    def color_node(n):
        is_final = G.nodes[n]['is_final']
        return 'blue' if is_final else 'green'

    def pos_node(n: VehicleState):
        w = -n.wait * 0.2
        return n.x, n.v + w

    node_size = 20
    node_color = [color_node(_) for _ in G.nodes]
    pos = {_: pos_node(_) for _ in G.nodes}
    r = Report(nid=player_name)

    with r.plot('one') as plt:
        nx.draw(G, pos=pos, node_color=node_color, cmap=plt.cm.Blues,
                node_size=node_size)
        plt.xlabel('x')
        plt.ylabel('v')
    logger.info('layout')
    pos = graphviz_layout(G, prog='dot')
    logger.info('drawing')
    with r.plot('s') as plt:
        nx.draw(G, pos=pos, node_color=node_color, cmap=plt.cm.Blues,
                node_size=node_size)
        plt.xlabel('x')
        plt.ylabel('v')

    return r


def igraph_from_nx(G: DiGraph):
    nx.write_graphml(G, 'graph.graphml')  # Export NX graph to file

    import igraph as ig
    Gix = ig.read('graph.graphml', format="graphml")  # Create new IG graph from file
    return Gix


def report_game(game: Game):
    G = get_game_graph(game)
    G2 = convert_node_labels_to_integers(G)
    for (n1, n2, d) in G2.edges(data=True):
        d.clear()
    nx.write_graphml(G2, 'game.graphml')
    r = Report(nid='game')
    logger.info('layout')
    pos = graphviz_layout(G, prog='dot')
    logger.info('drawing')

    def color_node(n):
        is_final1 = G.nodes[n]['is_final1']
        is_final2 = G.nodes[n]['is_final2']
        is_joint_final = G.nodes[n]['is_joint_final']
        if is_final1:
            return 'blue'
        if is_final2:
            return 'green'
        if is_joint_final:
            return 'red'
        return 'black'

    node_size = 20
    node_color = [color_node(_) for _ in G.nodes]

    with r.plot('s') as plt:
        nx.draw(G, pos=pos, node_color=node_color,
                cmap=plt.cm.Blues,
                node_size=node_size)
        plt.xlabel('x')
        plt.ylabel('v')
    return r


def get_game_graph(game: Game) -> DiGraph:
    players = game.players
    assert len(players) == 2
    p1, p2 = list(players)
    P1 = players[p1]
    P2 = players[p2]
    # G1 = get_player_graph(players[p1])
    # G2 = get_player_graph(players[p2])

    G = DiGraph()
    stack = []
    for n1, n2 in product(P1.initial, P2.initial):
        G.add_node((n1, n2), is_final2=False, is_final1=False, is_joint_final=False, is_initial=True, generation=0)
        stack.append((n1, n2))

    logger.info(stack=stack)
    i = 0
    while stack:
        print(i, len(stack), len(G.nodes))
        i += 1
        S = stack.pop(0)
        assert S in G.nodes

        n1, n2 = S

        if n1 is None or G.nodes[S]['is_final1']:
            succ1 = {None: {None}}
        else:
            succ1 = P1.dynamics.successors(n1)

        if n2 is None or G.nodes[S]['is_final2']:
            succ2 = {None: {None}}
        else:
            succ2 = P2.dynamics.successors(n2)

        generation = G.nodes[S]['generation']

        for (u1, s1s), (u2, s2s) in product(succ1.items(), succ2.items()):
            for s1, s2 in product(s1s, s2s):
                # check_isinstance(s1, VehicleState)
                # check_isinstance(s2, VehicleState)
                S2 = s1, s2
                if S2 not in G.nodes:
                    is_final1 = P1.personal_reward_structure.is_personal_final_state(s1) if s1 else True
                    is_final2 = P2.personal_reward_structure.is_personal_final_state(s2) if s2 else True

                    if s1 and s2:
                        is_joint_final = len(game.joint_reward.is_joint_final_state({p1: s1, p2: s2})) > 0
                    else:
                        is_joint_final = False
                    G.add_node(S2, is_final2=is_final2, is_final1=is_final1, is_joint_final=is_joint_final,
                               is_initial=False, generation=generation + 1)
                    if not (is_joint_final):
                        stack.append(S2)
                G.add_edge(S, S2, u1=u1, u2=u2)
                G.nodes[S2]['generation'] = min(G.nodes[S2]['generation'], generation + 1)

    generations = defaultdict(list)
    for n in G.nodes:
        g = G.nodes[n]['generation']
        others = generations[g]
        G.nodes[n]['y'] = len(others)
        G.nodes[n]['x'] = g
        others.append(n)

    return G
