"""Local-test compatibility for the course-only ``mugrade`` package."""

import sys
import types


try:
    import mugrade  # noqa: F401
except ModuleNotFoundError:
    mugrade = types.ModuleType("mugrade")
    mugrade.submit = lambda value: None
    sys.modules["mugrade"] = mugrade
