"""Pure computational-delay scaling for explicitly configured node profiles."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Chain.Node import Node


class ComputationalDelay:
    """Scale validation work relative to SymBChainSim's 20 GHz calibration.

    The 20 GHz reference preserves the simulator's existing validation delay.
    It is a SymBChainSim calibration convention, not a constant specified by
    Liu et al. Missing capabilities disable scaling for that node.
    """

    REFERENCE_CAPABILITY_GHZ = 20.0
    FORMULA = "scaled_delay = base_delay * 20 GHz / node_capability_ghz"

    @classmethod
    def scale(cls, base_delay: float, computational_capability_ghz: float | None) -> float:
        if computational_capability_ghz is None:
            return base_delay
        if computational_capability_ghz <= 0:
            raise ValueError("Computational capability must be positive")
        return base_delay * cls.REFERENCE_CAPABILITY_GHZ / computational_capability_ghz

    @classmethod
    def for_node(cls, base_delay: float, node: "Node") -> float:
        return cls.scale(base_delay, node.computational_capability_ghz)
