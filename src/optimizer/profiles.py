
from __future__ import annotations

from dataclasses import dataclass

ASSETS: tuple[str, ...] = ("SPY", "IEF", "LQD", "VNQ")
EQUITY_ASSETS: tuple[str, ...] = ("SPY", "VNQ")   # actions = SPY + VNQ
GOV_BOND_ASSETS: tuple[str, ...] = ("IEF",)       # obligations d'État


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    max_vol: float          # volatilité annualisée maximale du portefeuille
    max_equity: float       # plafond actions (avant ajustement crise)
    min_gov_bonds: float    # plancher obligations d'État


PROFILES: dict[str, Profile] = {
    "conservative": Profile("conservative", "Prudent", 0.06, 0.25, 0.30),
    "balanced": Profile("balanced", "Équilibré", 0.10, 0.50, 0.15),
    "aggressive": Profile("aggressive", "Dynamique", 0.15, 0.75, 0.00),
}

_ALIASES = {
    "prudent": "conservative", "conservateur": "conservative",
    "equilibre": "balanced", "équilibré": "balanced", "equilibré": "balanced",
    "dynamique": "aggressive", "agressif": "aggressive",
}


def get_profile(name: str | Profile) -> Profile:
    if isinstance(name, Profile):
        return name
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in PROFILES:
        valid = sorted(PROFILES) + sorted(_ALIASES)
        raise ValueError(f"Profil inconnu: {name!r}. Valeurs possibles: {valid}")
    return PROFILES[key]