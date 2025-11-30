"""
ICEBURG Etymologist Agent
Analyzes linguistic patterns and etymology
"""

from __future__ import annotations

from ..config import IceburgConfig


class EtymologistAgent:
    """
    Analyzes linguistic patterns and etymology
    """

    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
