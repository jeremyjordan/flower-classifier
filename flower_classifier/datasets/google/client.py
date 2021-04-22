import os

from google_images_search import GoogleImagesSearch


def get_secrets():
    required_keys = {"GIS_API_KEY", "GIS_PROJECT_CX"}

    if not required_keys.issubset(set(os.environ.keys())):
        missing_keys = required_keys.difference(set(os.environ.keys()))
        raise ValueError(f"Missing environment variables for: {missing_keys}")

    secrets = {}
    for k in required_keys:
        secrets[k] = os.environ[k]

    return secrets


def query_google_images(flower_name: str):
    secrets = get_secrets()
    gis = GoogleImagesSearch(secrets["GIS_API_KEY"], secrets["GIS_PROJECT_CX"])
    gis.search({"q": f"{prediction_name} flower", "num": 3})
    return [result.url for result in gis.results()]
