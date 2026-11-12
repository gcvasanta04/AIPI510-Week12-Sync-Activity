"""Central configuration: paths, column names, split settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# Optional local copy of the raw UCI file (adult.data, no header).
# If present it is used instead of downloading from OpenML.
LOCAL_RAW_PATH = DATA_DIR / "adult.data"

UCI_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

TARGET = "income"
POSITIVE_LABEL = ">50K"

# fnlwgt is a census sampling weight, not a person attribute;
# education is a duplicate of education_num.
DROP_COLUMNS = ["fnlwgt", "education"]

RANDOM_STATE = 42
TEST_SIZE = 0.2   # fraction of all data held out for final test
VAL_SIZE = 0.2    # fraction of the remaining data used for model selection
