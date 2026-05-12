from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, List

from .types import Component


@dataclass(order=True)
class _HeapItem:
    neg_size: int
    idx: int
    comp: Component


def split_component(comp: Component) -> list[Component]:
    t_len, c_len = comp.shape
    if t_len <= 1 and c_len <= 1:
        return []

    if t_len >= c_len and t_len > 1:
        mid = comp.t0 + t_len // 2
        return [
            Component(
                cid=f"{comp.cid}.0",
                t0=comp.t0,
                t1=mid,
                c0=comp.c0,
                c1=comp.c1,
                depth=comp.depth + 1,
                parent=comp.cid,
            ),
            Component(
                cid=f"{comp.cid}.1",
                t0=mid,
                t1=comp.t1,
                c0=comp.c0,
                c1=comp.c1,
                depth=comp.depth + 1,
                parent=comp.cid,
            ),
        ]

    if c_len > 1:
        mid = comp.c0 + c_len // 2
        return [
            Component(
                cid=f"{comp.cid}.0",
                t0=comp.t0,
                t1=comp.t1,
                c0=comp.c0,
                c1=mid,
                depth=comp.depth + 1,
                parent=comp.cid,
            ),
            Component(
                cid=f"{comp.cid}.1",
                t0=comp.t0,
                t1=comp.t1,
                c0=mid,
                c1=comp.c1,
                depth=comp.depth + 1,
                parent=comp.cid,
            ),
        ]

    return []


def split_component_time(comp: Component) -> list[Component]:
    t_len, _ = comp.shape
    if t_len <= 1:
        return []
    mid = comp.t0 + t_len // 2
    return [
        Component(
            cid=f"{comp.cid}.0",
            t0=comp.t0,
            t1=mid,
            c0=comp.c0,
            c1=comp.c1,
            depth=comp.depth + 1,
            parent=comp.cid,
        ),
        Component(
            cid=f"{comp.cid}.1",
            t0=mid,
            t1=comp.t1,
            c0=comp.c0,
            c1=comp.c1,
            depth=comp.depth + 1,
            parent=comp.cid,
        ),
    ]


def make_initial_partition(t_steps: int, channels: int, k0: int) -> list[Component]:
    if k0 < 1:
        raise ValueError("k0 must be >= 1")

    root = Component(cid="root", t0=0, t1=t_steps, c0=0, c1=channels)
    leaves: list[Component] = [root]
    heap: list[_HeapItem] = [_HeapItem(-root.n_points, 0, root)]
    seq = 1

    while len(leaves) < k0 and heap:
        item = heapq.heappop(heap)
        comp = item.comp
        children = split_component(comp)
        if not children:
            continue

        leaves.remove(comp)
        leaves.extend(children)
        for child in children:
            heapq.heappush(heap, _HeapItem(-child.n_points, seq, child))
            seq += 1

    return leaves


def make_initial_partition_time(t_steps: int, channels: int, k0: int) -> list[Component]:
    if k0 < 1:
        raise ValueError("k0 must be >= 1")
    root = Component(cid="root", t0=0, t1=t_steps, c0=0, c1=channels)
    leaves: list[Component] = [root]
    heap: list[_HeapItem] = [_HeapItem(-(root.t1 - root.t0), 0, root)]
    seq = 1

    while len(leaves) < k0 and heap:
        item = heapq.heappop(heap)
        comp = item.comp
        children = split_component_time(comp)
        if not children:
            continue
        leaves.remove(comp)
        leaves.extend(children)
        for child in children:
            heapq.heappush(heap, _HeapItem(-(child.t1 - child.t0), seq, child))
            seq += 1

    return leaves


def components_cost(components: Iterable[Component], total_points: int) -> float:
    return sum(c.n_points for c in components) / float(total_points)
