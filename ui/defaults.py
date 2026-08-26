"""UI-level defaults for layer composite visualization."""

from app_core import LayerInfo
from odb_cam_renderer import CompositeLayer


def default_spec(layer: LayerInfo) -> CompositeLayer:
    kind, name = layer.layer_type.upper(), layer.name.upper()
    if "DRILL" in kind or "DRILL" in name or name.startswith(("UV_", "TH_", "GDRILL_")):
        return CompositeLayer(layer.name, "REPLACE", 96)
    if "SOLDER_MASK" in kind or name.startswith("SM-"):
        return CompositeLayer(layer.name, "ADD", 160)
    if kind == "MIXED":
        return CompositeLayer(layer.name, "ADD", 220)
    return CompositeLayer(layer.name, "REPLACE", 255)
