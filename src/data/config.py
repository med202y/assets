import os
from pathlib import Path

from dotenv import load_dotenv


RACINE = Path(__file__).resolve().parents[2]


load_dotenv(RACINE / ".env")

FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise RuntimeError(
        "FRED_API_KEY introuvable. Crée un fichier .env à la racine du projet "
        "avec la ligne : FRED_API_KEY=ta_cle"
    )
