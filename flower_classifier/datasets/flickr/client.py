"""
See documentation for flickrapi here: https://stuvel.eu/flickrapi-doc/
"""

import io
import json
import logging

import flickrapi
from flickrapi.auth import FlickrAccessToken
from PIL import Image

from flower_classifier.datasets import get_secrets

logger = logging.getLogger(__name__)


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


def is_photo_duplicate(flickr_client: flickrapi.FlickrAPI, filename: str, user_id="192840150@N08"):
    """
    Returns a boolean value denoting whether or not the user already has a photo with the same filename.
    """
    results = flickr_client.photos_search(user_id=user_id, text=filename)
    photos = json.loads(results)["photos"]["photo"]
    if photos:
        for photo in photos:
            if photo["title"] == filename:
                return True
    return False


def upload_photo(flickr_client: flickrapi.FlickrAPI, filename: str, pil_image: Image = None, tags: set = None):
    """
    Uploads a photo to the authenticated user's Flickr account.

    If pil_image is not provided, the image is read from filename.
    """
    if is_photo_duplicate(flickr_client, filename):
        logger.info(f"Photo {filename} has already been uploaded, skipping...")
        return
    if pil_image is not None:
        fileobj = io.BytesIO()
        pil_image.save(fileobj, format="JPEG")
        fileobj.seek(0)
    else:
        fileobj = None
    tags = " ".join(tags) if tags else None
    _ = flickr_client.upload(filename=filename, fileobj=fileobj, tags=tags, format="etree")
