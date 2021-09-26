"""
See documentation for flickrapi here: https://stuvel.eu/flickrapi-doc/
"""

import io
import json
import logging
import urllib.request
from pathlib import Path
from typing import List

import flickrapi
from flickrapi.auth import FlickrAccessToken
from PIL import Image

from flower_classifier.datasets import get_secrets
from flower_classifier.datasets.flickr.auth import USER_ID

logger = logging.getLogger(__name__)


def get_authenticated_client(format="json"):
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
    flickr_client = flickrapi.FlickrAPI(secrets["API_KEY"], secrets["API_SECRET"], token=auth_token, format=format)
    return flickr_client


def is_photo_duplicate(flickr_client: flickrapi.FlickrAPI, filename: str, user_id=USER_ID):
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


def get_photo_info(flickr_client: flickrapi.FlickrAPI, photo_id: str):
    result = flickr_client.do_flickr_call(_method_name="flickr.photos.getInfo", photo_id=photo_id, format="json")
    photo_info = json.loads(result).get("photo")
    return photo_info


def get_photo_tags(flickr_client: flickrapi.FlickrAPI, photo_id: str):
    photo_info = get_photo_info(flickr_client=flickr_client, photo_id=photo_id)
    existing_tags = [t["raw"] for t in photo_info.get("tags", {}).get("tag", [])]
    tags = set(existing_tags)
    return tags


def update_photo_tags(
    flickr_client: flickrapi.FlickrAPI, photo_id: str, insert_tags: List[str] = None, remove_tags: List[str] = None
):
    if insert_tags is None:
        insert_tags = []
    if remove_tags is None:
        remove_tags = []

    tags = get_photo_tags(flickr_client=flickr_client, photo_id=photo_id)
    tags.update(insert_tags)
    tags.difference_update(remove_tags)
    new_tags = " ".join(tags) if tags else None
    flickr_client.do_flickr_call(_method_name="flickr.photos.setTags", photo_id=photo_id, tags=new_tags)


def download_photo(flickr_client: flickrapi.FlickrAPI, photo_id: str, download_path: Path):
    result = flickr_client.do_flickr_call(_method_name="flickr.photos.getSizes", photo_id=photo_id, format="json")
    photo_sizes = json.loads(result)["sizes"]["size"]
    photo_url = None
    for size in reversed(photo_sizes):
        if size["label"] == "Original":
            photo_url = size["source"]
            break
    if photo_url:
        urllib.request.urlretrieve(photo_url, download_path)
