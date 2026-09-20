"""Shared construction of immutable Liu runtime snapshots from simulator parameters."""

from Chain.Consensus.LiuRuntime.Common.EpochContext import LiuRuntimeEpochContext
from Liu.EpochConfiguration import EpochConfiguration
from Liu.LinkState import LinkStateMatrix
from Liu.Threat import ThreatScenario
from Parameters import Parameters


def configured_link_state(config: dict, node_count: int) -> LinkStateMatrix:
    if "directed_link_rates_mbps" in config:
        links = LinkStateMatrix.from_rows(config["directed_link_rates_mbps"])
        if links.node_count != node_count:
            raise ValueError("directed_link_rates_mbps must be an exact Nn-by-Nn matrix")
        return links
    rate = float(config["fixed_link_rate_mbps"])
    return LinkStateMatrix.from_rows(
        tuple(tuple(None if sender == receiver else rate for receiver in range(node_count)) for sender in range(node_count))
    )


def runtime_epoch_from_parameters(node, protocol, config: dict) -> LiuRuntimeEpochContext:
    node_count = Parameters.application["Nn"]
    epoch = EpochConfiguration(
        epoch_id=node.reconfiguration_state.confchain[-1].depth,
        validator_set=node.active_validator_set,
        consensus_protocol=protocol,
        block_size_mb=node.reconfiguration_state.configuration.block_size,
        block_interval_s=node.reconfiguration_state.configuration.block_time,
    )
    return LiuRuntimeEpochContext(
        epoch,
        Parameters.node_profile_set,
        configured_link_state(config, node_count),
        ThreatScenario(node_count, tuple(config.get("malicious_node_ids", ()))),
    )
