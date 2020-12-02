from os.path import join
import math
import networkx as nx
from networkx import MultiDiGraph
from reprep import Report
from time import perf_counter
from typing import List, Tuple

from trajectory_game import VehicleState, VehicleGeometry, World, SplineTransitionPath, BicycleDynamics, AllTrajectories


def report_trajectories(G: MultiDiGraph) -> Report:
    r = Report(nid="trajectories")
    caption = "Trajectories generated"

    def pos_node(n: VehicleState):
        x = G.nodes[n]["x"]
        y = G.nodes[n]["y"]
        return float(x), float(y)

    def line_width(n: VehicleState):
        return float(1. / pow(2., G.edges[n]['gen']))

    def node_sizes(n: VehicleState):
        return float(1. / pow(2., G.nodes[n]['gen']))

    pos = {_: pos_node(_) for _ in G.nodes}
    widths = [line_width(_) for _ in G.edges]
    # node_size = [node_sizes(_) for _ in G.nodes]

    with r.plot("s", caption=caption) as plt:
        # nx.draw_networkx_nodes(
        #     G,
        #     pos=pos,
        #     nodelist=G.nodes(),
        #     cmap=plt.cm.Blues,
        #     node_size=node_size,
        # )
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=G.edges(),
            alpha=0.5,
            arrows=False,
            width=widths,
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('auto')
    return r


def create_world() -> World:
    s: List[float] = [float(_) for _ in range(10)]
    x: List[float] = [0. for _ in range(10)]
    y: List[float] = s
    ref = SplineTransitionPath[float](s=s, x=x, y=y, order=3)
    left: List[Tuple[float, float]] = [(-5., float(_)) for _ in range(10)]
    right: List[Tuple[float, float]] = [(5., float(_)) for _ in range(10)]
    world = World(ref_path=ref, left_xy=left, right_xy=right)
    return world


def test_dynamics():

    max_gen = 2
    dt = .3
    steps_dst, step_dst = 3, math.pi / 6
    steps_acc, step_acc = 3, 4.
    u_acc = frozenset([_ * step_acc for _ in range(-steps_acc//2+1, steps_acc//2+1)])
    u_dst = frozenset([_ * step_dst for _ in range(-steps_dst//2+1, steps_dst//2+1)])
    vg = VehicleGeometry(w=1., lf=1., lr=1.)
    dynamics = BicycleDynamics(v_max=15., v_min=5., st_max=math.pi / 4,
                               vg=vg, u_acc=u_acc, u_dst=u_dst)
    state_init = VehicleState(x=0., y=0., th=math.pi/2, v=10., st=0., t=0.)
    stack = list([state_init])
    G = MultiDiGraph()

    def add_node(s, gen):
        G.add_node(s, gen=gen, x=s.x, y=s.y)

    add_node(state_init, gen=0)
    i: int = 0
    expanded = set()
    tic = perf_counter()
    while stack:
        i += 1
        s1 = stack.pop(0)
        assert s1 in G.nodes
        if s1 in expanded:
            continue
        n_gen = G.nodes[s1]['gen']
        expanded.add(s1)
        successors = dynamics.successors(s1, dt)
        for u, s2 in successors.items():
            if s2 not in G.nodes:
                add_node(s2, gen=n_gen+1)
                if n_gen + 1 < max_gen:
                    stack.append(s2)
            G.add_edge(s1, s2, u=u, gen=n_gen)

    toc = perf_counter() - tic
    print('Trajectory graph generation time = {} s'.format(toc))

    tic = perf_counter()
    ego_name = 'Car1'
    all_traj = AllTrajectories(G=G, ego_name=ego_name)
    toc = perf_counter() - tic
    print('Trajectory list generation time = {} s'.format(toc))

    tic = perf_counter()
    world = create_world()
    toc = perf_counter() - tic
    print('World generation time = {} s'.format(toc))

    tic = perf_counter()
    result = all_traj.evaluate_trajectories(world=world)
    toc = perf_counter() - tic
    print('Metric evaluation time = {} s'.format(toc))

    direc = "out/tests/"
    # report = report_trajectories(G=G)
    # report.to_html(join(direc, "trajectories.html"))



