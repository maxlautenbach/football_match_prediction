"""Season-outlook Kicktipp tips (Meister / Herbstmeister / Bottom-3 / Torjäger)."""

from models.saison_ausblick.model import SaisonAusblickModel
from models.saison_ausblick.train import train as train_saison_ausblick

MODEL_TYPE = "saison_ausblick"

__all__ = ["MODEL_TYPE", "SaisonAusblickModel", "train_saison_ausblick"]
