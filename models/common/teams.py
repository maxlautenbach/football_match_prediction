"""Team name normalization shared by all model types."""

from __future__ import annotations

import math


def normalize_team_name(name: str) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    s = str(name).strip()
    s = " ".join(s.split())
    s = s.replace(". ", ".")
    s = s.replace("\u2019", "'")
    s = s.replace("\u2013", "-")
    s = s.replace("\u2014", "-")
    return s
