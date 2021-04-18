import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DATA_DIR = "/content/drive/My Drive/Flowers/"
REPO_DIR = Path(__file__).parent.parent


if Path(REPO_DIR / "secrets.json").exists():
    logger.info("Found secrets file, populating values as environment variables.")
    secrets = json.loads((REPO_DIR / "secrets.json").read_text())
    for k, v in secrets.items():
        os.environ[k] = v
