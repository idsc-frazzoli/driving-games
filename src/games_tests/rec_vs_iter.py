import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Set, Mapping

from dg_commons.time import time_function

TERMINAL = 1000
sys.setrecursionlimit(TERMINAL**2)
ACTIONS = (2, 9, 14, 17)


@dataclass(frozen=True)
class FakeNode:
    js: Any
    value_1: Any
    value_2: Any
    value_3: Any


@dataclass
class ContextRec:
    depth = 0
    processed: Set = field(default_factory=set)
    nodes: Mapping[Any, FakeNode] = field(default_factory=dict)


@dataclass
class ContextIter(ContextRec):
    to_expand: Any = field(default_factory=deque)


@time_function
def build_gg_rec(init_js=1):
    ctx = ContextRec()
    _dynamics_rec(ctx, init_js)
    return ctx


def _dynamics_rec(ctx, js):
    if js in ctx.processed or js > TERMINAL:
        return
    # ic2 = replace(ctx, depth=ctx.depth + 1)

    for action in ACTIONS:
        new_js = js + action

        _dynamics_rec(ctx, new_js)

    fn = FakeNode(js, js**3, js % 2, js / 10)
    ctx.processed.add(js)
    ctx.nodes[js] = fn
    return


@time_function
def build_gg_iter(init_js=1):
    ctx = ContextIter()
    ctx.to_expand.append(init_js)
    while ctx.to_expand:
        js_next = ctx.to_expand.popleft()
        _dynamics_iter(ctx, js_next)
    return ctx


def _dynamics_iter(ctx, js):
    if js in ctx.processed or js > TERMINAL:
        return
    for action in ACTIONS:
        new_js = js + action
        if new_js not in ctx.processed:
            ctx.to_expand.append(new_js)
    fn = FakeNode(js, js**3, js % 2, js / 10)
    ctx.processed.add(js)
    ctx.nodes[js] = fn
    return


# @dataclass(order=True)
# class PrioritizedJointState:
#     priority: int
#     js: Any = field(compare=False)


if __name__ == "__main__":
    ic_rec = build_gg_rec()
    ic_iter = build_gg_iter()

    # logger.info(f'ic_iter.nodes:', nodes=ic_iter.nodes)
    # logger.info(f'ic_rec.nodes:', nodes=ic_rec.nodes)

    assert ic_iter.nodes == ic_rec.nodes, "ic_iter.nodes != ic_rec.nodes"
