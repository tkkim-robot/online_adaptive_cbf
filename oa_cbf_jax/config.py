"""Shared config implementation."""

from dataclasses import asdict, dataclass

import hashlib

import json

import math

@dataclass(frozen=True)
class UnicycleConfig:
    dt: float = 0.05
    radius: float = 0.3
    a_max: float = 0.5
    w_max: float = 0.5
    v_max: float = 1.0
    clearance_buffer: float = 0.05
    cbf_margin: float = 0.0
    qp_tolerance: float = 1e-5
    goal_tolerance: float = 0.25
    integration_substeps: int = 4
    guidance_horizon: float = 0.0
    guidance_min_speed: float = 0.0
    guidance_kernel: str = 'scan'
    guidance_goal_braking: bool = False
    guidance_soft_clearance: bool = False
    guidance_route_preview: bool = False
    guidance_wide_turns: bool = False
    guidance_detour: bool = False
    guidance_turn_return: bool = False
    # False preserves historical experiments; static benchmark manifests must
    # explicitly set True. Sensor velocity error then never moves the plant.
    stationary_obstacles: bool = False

    def __post_init__(self):
        if not isinstance(self.stationary_obstacles, bool):
            raise ValueError('Stationary-obstacle contract must be boolean')
        if not all(math.isfinite(value) for key,value in asdict(self).items() if key!='guidance_kernel'):
            raise ValueError("Configuration must be finite")
        for key in ("dt", "radius", "a_max", "w_max", "v_max", "qp_tolerance", "goal_tolerance"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if isinstance(self.integration_substeps,bool) or not isinstance(self.integration_substeps,int) or self.integration_substeps < 1 or self.clearance_buffer < 0:
            raise ValueError("Invalid integration or geometry configuration")
        if not 0 <= self.guidance_horizon <= 12:
            raise ValueError("Guidance horizon must be between zero (disabled) and 12 seconds")
        if not 0 <= self.guidance_min_speed <= self.v_max:
            raise ValueError("Guidance cruise target must respect the robot speed limit")
        if self.guidance_kernel not in ('scan','vectorized') or not isinstance(self.guidance_goal_braking,bool):
            raise ValueError('Invalid guidance implementation or goal braking option')
        if not isinstance(self.guidance_soft_clearance,bool):
            raise ValueError('Guidance clearance preference must be boolean')
        if not isinstance(self.guidance_route_preview,bool):
            raise ValueError('Route preview option must be boolean')
        if not isinstance(self.guidance_wide_turns,bool):
            raise ValueError('Wide-turn guidance option must be boolean')
        if not isinstance(self.guidance_detour,bool):
            raise ValueError('Local detour option must be boolean')
        if not isinstance(self.guidance_turn_return,bool):
            raise ValueError('Turn-and-return guidance option must be boolean')

def config_hash(config):
    data = asdict(config) if hasattr(config, "__dataclass_fields__") else config
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
