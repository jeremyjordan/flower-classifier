import io

import flickrapi
from flickrapi.auth import FlickrAccessToken
from PIL import Image

from flower_classifier.datasets.flickr.auth import get_secrets


def get_authenticated_client():
    """
    Returns a Flickr API client which authenticates using:
        - API keys to authenticate at the application level
        - OAuth token to authenticate at the user level
    """
    secrets = get_secrets()

    auth_token = FlickrAccessToken(
        token=secrets["OAUTH_TOKEN"],
        token_secret=secrets["OAUTH_TOKEN_SECRET"],
        access_level=secrets["OAUTH_ACCESS_LEVEL"],
        fullname=secrets["OAUTH_FULL_NAME"],
        username=secrets["OAUTH_USERNAME"],
        user_nsid=secrets["OAUTH_USER_NSID"],
    )
    flickr_client = flickrapi.FlickrAPI(secrets["API_KEY"], secrets["API_SECRET"], token=auth_token, format="json")
    return flickr_client


def upload_photo(flickr_client: flickrapi.FlickrAPI, filename: str, pil_image: Image = None):
    """
    Uploads a photo to the authenticated user's Flickr account.

    If pil_image is not provided, the image is read from filename.
    """
    print(pil_image.format)
    if pil_image is not None:
        fileobj = io.BytesIO()
        pil_image.save(fileobj, format="JPEG")
        fileobj.seek(0)
    else:
        fileobj = None
    flickr_client.upload(filename=filename, fileobj=fileobj, format="etree")
