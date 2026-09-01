"""Provider-neutral tool routing for Yuki's two runtime architectures.

The live :mod:`tools` registry remains the source of truth.  This package only
adds strict dispatcher-facing schemas, call normalization, typed binding, and
the optional dedicated-dispatcher client.
"""

from .core import (
    AUTONOMOUS_DECISION_SCHEMA,
    AUTONOMOUS_DECISION_SYSTEM,
    DEFAULT_DISPATCHER_MODEL,
    ROUTING_MODES,
    ROUTING_PROTOCOLS,
    RoutingRegistry,
    build_main_brain_decision_prompt,
    normalize_routing_mode,
    normalize_routing_protocol,
    parse_autonomous_decision,
    parse_main_brain_decision,
    referenced_literal_sources,
)
from .runtime import DedicatedDispatcherClient

__all__ = [
    "AUTONOMOUS_DECISION_SCHEMA",
    "AUTONOMOUS_DECISION_SYSTEM",
    "DEFAULT_DISPATCHER_MODEL",
    "ROUTING_MODES",
    "ROUTING_PROTOCOLS",
    "DedicatedDispatcherClient",
    "RoutingRegistry",
    "build_main_brain_decision_prompt",
    "normalize_routing_mode",
    "normalize_routing_protocol",
    "parse_autonomous_decision",
    "parse_main_brain_decision",
    "referenced_literal_sources",
]
