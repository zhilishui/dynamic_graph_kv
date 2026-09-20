"""Topology-generator boundary for request-specific graph traces."""

from graphkv.topology.model import GraphSpec, graph_identity
from graphkv.topology.trace import TopologyRequest, load_topology_trace

__all__ = [
    "GraphSpec",
    "TopologyRequest",
    "graph_identity",
    "load_topology_trace",
]
