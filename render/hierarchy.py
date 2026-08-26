"""Stable rendering API for hierarchy-aware ODB rendering.

The implementation remains in ``hierarchy_renderer`` during the compatibility
phase so existing callers keep working while new code imports from ``render``.
"""

from hierarchy_renderer import (
    CacheSnapshot,
    FastODBRenderer,
    StepFrame,
    StepInstance,
    adaptive_preview_dpi,
    cache_snapshot,
    cached_profile,
    cached_repeats,
    cached_step_frame,
    clear_render_caches,
)

__all__ = [
    "CacheSnapshot",
    "FastODBRenderer",
    "StepFrame",
    "StepInstance",
    "adaptive_preview_dpi",
    "cache_snapshot",
    "cached_profile",
    "cached_repeats",
    "cached_step_frame",
    "clear_render_caches",
]
