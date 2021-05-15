import json
import logging
from pathlib import Path
from typing import List

from google_images_search import GoogleImagesSearch
from tqdm import tqdm

from flower_classifier import REPO_DIR
from flower_classifier.datasets import get_secrets
from flower_classifier.datasets.oxford_flowers import NAMES as oxford_breeds

logger = logging.getLogger(__name__)

RESULTS_CACHE_FILE = Path(REPO_DIR, "assets", "breed_examples.json")
global RESULTS_CACHE
RESULTS_CACHE = json.loads(RESULTS_CACHE_FILE.read_text())


def query_google_images(flower_name: str):
    if flower_name in RESULTS_CACHE:
        return RESULTS_CACHE[flower_name]
    else:
        logger.info(f"Cache miss for {flower_name}, querying Google Images Search...")
        secrets = get_secrets()
        gis = GoogleImagesSearch(secrets["GIS_API_KEY"], secrets["GIS_PROJECT_CX"])
        gis.search({"q": f"{flower_name} flower", "num": 3})
        return [result.url for result in gis.results()]


def export_similar_images(breeds: List[str] = oxford_breeds, query_limit: int = 100):
    query_count = 0
    for breed in tqdm(breeds):
        if breed not in RESULTS_CACHE:
            urls = query_google_images(breed)
            query_count += 1
            RESULTS_CACHE[breed] = urls
        if query_count >= query_limit:
            break
    RESULTS_CACHE_FILE.write_text(json.dumps(RESULTS_CACHE))
