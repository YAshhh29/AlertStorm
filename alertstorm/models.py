# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Alertstorm Environment.
"""

from typing import List, Dict, Optional, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State

class AlertstormAction(Action):
    """
    Action for the Alertstorm environment.
    The agent investigates logs, suppresses noise, or proposes a root cause.    
    """
    action_type: str = Field(..., description="Must be one of: 'investigate', 'suppress_alert', 'propose_root_cause'")
    targets: List[str] = Field(..., description="List of service names or alert_ids to target. Single items should be a 1-element list.")
    confidence: Optional[float] = Field(None, description="Confidence level when proposing a root cause (0.0 to 1.0).")

class AlertstormObservation(Observation):
    """
    Observation from the Alertstorm environment.
    What the agent sees at 3:00 AM (The PagerDuty interface).
    """
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="List of currently firing alerts. Keys: id, service, type, timestamp.")       
    dependency_graph: Dict[str, List[str]] = Field(default_factory=dict, description="Topology mapping. Keys: service_name. Values: List of services it depends on.")
    recent_logs: Optional[str] = Field(None, description="Detailed logs returned if the previous action was 'investigate'.")

class AlertstormState(State):
    """
    State for the Alertstorm environment.
    Hidden tracking variables (Tasks, Steps, Secret Root Cause).
    """
    task_level: str = Field("standard_easy", description="The difficulty level of the current incident (standard_easy, standard_medium, standard_hard, enterprise_easy, enterprise_medium, enterprise_hard).")
    secret_root_causes: List[str] = Field(default_factory=list, description="HIDDEN: The true failing node(s). The agent does not see this.")
    time_elapsed: int = Field(0, description="Virtual minutes passed since incident start.")
    suppressed_alerts: List[str] = Field(default_factory=list, description="IDs of alerts the agent has muted.")
    active_alerts: List[Dict[str, Any]] = Field(default_factory=list, description="List of currently firing alerts.")
    rewarded_nodes: List[str] = Field(default_factory=list, description="Nodes already rewarded for partial progress.")

