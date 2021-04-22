from google_images_search import GoogleImagesSearch

from flower_classifier.datasets import get_secrets


def query_google_images(flower_name: str):
    secrets = get_secrets()
    gis = GoogleImagesSearch(secrets["GIS_API_KEY"], secrets["GIS_PROJECT_CX"])
    gis.search({"q": f"{flower_name} flower", "num": 3})
    return [result.url for result in gis.results()]
