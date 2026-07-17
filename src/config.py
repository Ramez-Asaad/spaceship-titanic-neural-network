"""Project paths and the hyperparameters documented in the original report."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

TARGET = "Transported"
ID = "PassengerId"

RANDOM_SEED = 42

# Hyperparameters exactly as specified in the Fall 24/25 report (Step 4).
REPORT_HPARAMS = {
    "learning_rate": 0.001,
    "hidden_units": 128,
    "hidden_layers": 3,
    "batch_size": 32,
    "epochs": 1000,
    "dropout_rate": 0.2,
    "hidden_activation": "relu",
    "output_activation": "sigmoid",
    "loss": "binary_crossentropy",
    "initializer": "xavier",
}

# The report selected only these raw columns, dropping HomePlanet, Name and all
# five spending columns. See README "What the original got wrong".
REPORT_COLUMNS = ["PassengerId", "CryoSleep", "Cabin", "Destination", "Age", "VIP"]

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

DECK_MAP = {d: i for i, d in enumerate("ABCDEFGT")}
DECK_MAP["Unknown"] = len(DECK_MAP)
SIDE_MAP = {"P": 0, "S": 1, "Unknown": 2}
DESTINATION_MAP = {
    "55 Cancri e": 0,
    "PSO J318.5-22": 1,
    "TRAPPIST-1e": 2,
    "Unknown": 3,
}
HOMEPLANET_MAP = {"Earth": 0, "Europa": 1, "Mars": 2, "Unknown": 3}
