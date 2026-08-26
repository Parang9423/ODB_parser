"""Backward-compatible import wrapper for the AOI ERT parser."""

from aoi.ert import ERTMetadata, parse_ert

__all__ = ["ERTMetadata", "parse_ert"]
