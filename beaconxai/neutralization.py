from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .types import Array, Component

NeutralizationMode = Literal["zero", "mean", "interp"]


@dataclass
class Neutralizer:
    mode: NeutralizationMode = "zero"
    channel_means: Array | None = None

    def __post_init__(self) -> None:
        if self.mode == "mean" and self.channel_means is None:
            raise ValueError("channel_means required for mode='mean'")

    def __call__(self, x: Array, components: Iterable[Component]) -> Array:
        z = x.copy()
        for comp in components:
            self._apply_component(z, comp)
        return z

    def _apply_component(self, z: Array, comp: Component) -> None:
        if self.mode == "zero":
            z[comp.t0 : comp.t1, comp.c0 : comp.c1] = 0.0
            return

        if self.mode == "mean":
            means = self.channel_means[comp.c0 : comp.c1]
            z[comp.t0 : comp.t1, comp.c0 : comp.c1] = means
            return

        if self.mode == "interp":
            self._interpolate(z, comp)
            return

        raise ValueError(f"unknown neutralization mode: {self.mode}")

    @staticmethod
    def _interpolate(z: Array, comp: Component) -> None:
        t0, t1, c0, c1 = comp.t0, comp.t1, comp.c0, comp.c1
        t_steps = z.shape[0]
        left = max(t0 - 1, 0)
        right = min(t1, t_steps - 1)

        if left == right:
            z[t0:t1, c0:c1] = z[left, c0:c1]
            return

        span = max(right - left, 1)
        left_val = z[left, c0:c1]
        right_val = z[right, c0:c1]

        for t in range(t0, t1):
            alpha = (t - left) / span
            z[t, c0:c1] = (1.0 - alpha) * left_val + alpha * right_val
