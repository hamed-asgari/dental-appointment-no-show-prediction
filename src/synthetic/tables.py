"""Public container type for generated Version 2 synthetic raw tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SyntheticTables:
    """Generated raw tables that exclude every hidden latent variable."""

    patients: pd.DataFrame
    dentists: pd.DataFrame
    appointments: pd.DataFrame


__all__ = ("SyntheticTables",)
