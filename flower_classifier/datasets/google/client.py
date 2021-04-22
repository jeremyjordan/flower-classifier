import os
from io import BytesIO

import PIL
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


def result_2_pil(image) -> PIL.Image:
    bytes_io = BytesIO()
    bytes_io.seek(0)
    image.copy_to(bytes_io)
    bytes_io.seek(0)
    try:
        return PIL.Image.open(bytes_io)
    except PIL.UnidentifiedImageError:
        return None


def query_google_images(prediction_name: str):
    secrets = get_secrets()
    gis = GoogleImagesSearch(secrets["GIS_API_KEY"], secrets["GIS_PROJECT_CX"])

    gis.search({"q": f"{prediction_name} flower", "num": 3})
    images = [result_2_pil(image) for image in gis.results()]
    return [image for image in images if image]
