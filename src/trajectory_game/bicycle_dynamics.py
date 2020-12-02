import itertools
import math
from os.path import join
from typing import FrozenSet, Set, Mapping, List
import networkx as nx
from networkx import MultiDiGraph
from reprep import Report
from time import perf_counter

from .structures import VehicleState, VehicleActions, VehicleGeometry


class BicycleDynamics:
    v_max: float
    """ Maximum speed [m/s] """

    v_min: float
    """ Minimum speed [m/s] """

    st_max: float
    """ Maximum steering angle [rad] """

    vg: VehicleGeometry
    """ The vehicle's geometry parameters"""

    u_acc: FrozenSet[float]
    """ Possible values of acceleration [m/s2] """

    u_dst: FrozenSet[float]
    """ Possible values of steering rate [rad/s] """

    def __init__(self, v_max: float, v_min: float, st_max: float,
                 vg: VehicleGeometry, u_acc: FrozenSet[float],
                 u_dst: FrozenSet[float]):
        self.v_max = v_max
        self.v_min = v_min
        self.st_max = st_max
        self.vg = vg
        self.u_acc = u_acc
        self.u_dst = u_dst

    def all_actions(self) -> Set[VehicleActions]:
        res = set()
        for acc, dst in itertools.product(self.u_acc, self.u_dst):
            res.add(VehicleActions(acc=acc, dst=dst))
        return res

    def successors(self, x: VehicleState, dt: float) -> Mapping[VehicleActions, VehicleState]:
        """ For each state, returns a dictionary U -> Possible Xs """

        # only allow inputs with feasible final vel, st
        u_acc = [_ for _ in self.u_acc if self.v_min <= x.v + _ * dt <= self.v_max]
        u_dst = [_ for _ in self.u_dst if -self.st_max <= x.st + _ * dt <= self.st_max]

        res = {}
        for acc, dst in itertools.product(u_acc, u_dst):
            u = VehicleActions(acc=acc, dst=dst)
            res[u] = self.successor(x, u, dt)
        return res

    def successor(self, x0: VehicleState, u: VehicleActions, dt: float):
        def clip(value, low, high):
            return max(low, min(high, value))

        vf = clip(x0.v + u.acc * dt, low=self.v_min, high=self.v_max)
        stf = clip(x0.st + u.dst * dt, low=-self.st_max, high=self.st_max)
        u_clip = VehicleActions(acc=(vf-x0.v)/dt, dst=(stf-x0.st)/dt)

        alpha = 1.
        k1 = self.dynamics(x0, u_clip)
        k2 = self.dynamics(x0 + k1 * (dt * alpha), u_clip)
        ret = x0 + k1 * (dt * (1 - 0.5 / alpha)) + k2 * (dt * (0.5 / alpha))
        return ret

    def dynamics(self, x0: VehicleState, u: VehicleActions) -> VehicleState:

        dx = x0.v
        dr = dx * math.tan(x0.st) / (self.vg.lf + self.vg.lr)
        dy = dr * self.vg.lr
        costh = math.cos(x0.th + dr/2.)
        sinth = math.sin(x0.th + dr/2.)

        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        ret = VehicleState(x=xdot, y=ydot, th=dr, v=u.acc, st=u.dst, t=1.)
        return ret


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


def test_dynamics():

    max_gen = 3
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
    print('Total trajectory generation time = {} s'.format(toc))
    direc = "out/tests/"
    report = report_trajectories(G=G)
    report.to_html(join(direc, "trajectories.html"))
