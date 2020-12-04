from os.path import join
import math
import networkx as nx
from networkx import MultiDiGraph
from reprep import Report
from time import perf_counter

from trajectory_game import VehicleState, VehicleGeometry, \
    BicycleDynamics, AllTrajectories


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


def generate_trajectory_graph() -> MultiDiGraph:

    max_gen = 2
    dt = .3
    steps_dst, step_dst = 3, math.pi / 6
    steps_acc, step_acc = 3, 4.
    u_acc = frozenset([_ * step_acc for _ in
                       range(-steps_acc//2+1, steps_acc//2+1)])
    u_dst = frozenset([_ * step_dst for _ in
                       range(-steps_dst//2+1, steps_dst//2+1)])
    vg = VehicleGeometry(w=1., lf=1., lr=1.)
    dynamics = BicycleDynamics(v_max=15., v_min=5., st_max=math.pi / 4,
                               vg=vg, u_acc=u_acc, u_dst=u_dst)
    state_init = VehicleState(x=0., y=0., th=math.pi/2,
                              v=10., st=0., t=0.)
    rect = dynamics.get_shared_resources(x=state_init)
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
    direc = "out/tests/"
    # report = report_trajectories(G=G)
    # report.to_html(join(direc, "trajectories.html"))
    return G


def trajectory_graph_to_list(G: MultiDiGraph) -> AllTrajectories:
    tic = perf_counter()
    ego_name = 'Car1'
    all_traj = AllTrajectories(G=G, ego_name=ego_name)
    toc = perf_counter() - tic
    print('Trajectory list generation time = {} s'.format(toc))
    return all_traj
