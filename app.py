#!/usr/bin/env python3
"""ODB CAM Viewer application entrypoint.

The implementation lives under ``ui``. This file intentionally stays thin so
headless AOI/ODB services can be imported without pulling the GUI together.
"""

from ui.defaults import default_spec
from ui.main_window import App

__all__ = ["App", "default_spec"]


if __name__ == "__main__":
    App().mainloop()
