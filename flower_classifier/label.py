import json
import shutil
from pathlib import Path

import flickrapi
import typer

from flower_classifier.datasets.flickr.auth import USER_ID as FLICKR_USER_ID
from flower_classifier.datasets.flickr.client import (
    download_photo,
    get_authenticated_client,
    get_photo_tags,
    update_photo_tags,
)

app = typer.Typer()

LABELED_TAG = "status:labeled"
UNCERTAIN_TAG = "status:uncertain"


def _confirm_directory_is_empty(dir: Path, exit_message="Exiting program."):
    if dir.exists() and any(dir.iterdir()):
        overwrite = typer.confirm(f"{dir} is not empty, are you sure you want to overwrite it?")
        if not overwrite:
            typer.echo(exit_message)
            raise typer.Exit()


def _parse_tags(flickr_client: flickrapi.FlickrAPI, photo_id: str):
    tags = get_photo_tags(flickr_client, photo_id)
    tags = [t.split(":") for t in tags if ":" in t]
    return dict(tags)


@app.command()
def get_photos(
    photo_count: int = typer.Option(50, prompt=True),
    label_dir: str = typer.Option(str(Path.home() / "flower_classifier_labels"), prompt=True),
):
    """
    Download a specified number of photos to be labeled.
    """
    # intialize directories
    label_dir = Path(label_dir)
    unlabeled_dir: Path = label_dir / "unlabeled"
    labeled_dir: Path = label_dir / "labeled"

    _confirm_directory_is_empty(
        unlabeled_dir,
        exit_message="Exiting program. Please return existing photos to the labeling pool and start again.",
    )
    _confirm_directory_is_empty(
        labeled_dir,
        exit_message="Exiting program. Make sure existing labeled photos are uploaded to Google Drive, "
        "empty the local folder (so you don't accidentally upload photos twice), and start again.",
    )

    unlabeled_dir.mkdir(exist_ok=True, parents=True)
    labeled_dir.mkdir(exist_ok=True, parents=True)

    # grab 50 unlabeled photos from flickr
    # and download to expected folder structure
    # TODO parallelize this with concurrent futures
    flickr_client = get_authenticated_client(format="etree")
    count = 0
    downloaded_photos = {}
    for count, photo in enumerate(
        flickr_client.walk(
            tag_mode="all",
            # adding a "-" in front of a tag excludes results that match the tag
            tags=f"-{LABELED_TAG}, -{UNCERTAIN_TAG}",
            user_id=FLICKR_USER_ID,
        )
    ):
        if count >= photo_count:
            break

        photo_id = photo.get("id")
        filename = photo.get("title")
        tags = _parse_tags(flickr_client, photo_id)
        user_judgement = tags.get("user_judgement", "unsure")
        predicted_breed = tags.get("pred", "unknown")
        download_path = unlabeled_dir / user_judgement / predicted_breed / filename
        download_path.parent.mkdir(parents=True, exist_ok=True)
        download_photo(flickr_client, photo_id, download_path)
        downloaded_photos[str(download_path)] = {"photo_id": photo_id, "tags": tags}

        # mark photo as labeled
        update_photo_tags(flickr_client, photo_id, insert_tags=[LABELED_TAG])

    (unlabeled_dir / "info.json").write_text(json.dumps(downloaded_photos))


@app.command()
def return_photos_to_labeling_pool(
    label_dir: str = typer.Option(str(Path.home() / "flower_classifier_labels"), prompt=True),
    remaining_uncertain: bool = typer.Option(False, help="Mark remaining photos as uncertain?", prompt=True),
):
    """
    Remove the "labeled" tag from any remaining photos in the unlabeled directory.
    If we need further review for remaining photos, tag as uncertain.
    """

    # intialize directories
    label_dir = Path(label_dir)
    unlabeled_dir: Path = label_dir / "unlabeled"
    downloaded_photos = json.loads((unlabeled_dir / "info.json").read_text())

    # remove the "labeled" tag from any remaining photos
    flickr_client = get_authenticated_client()

    for photo in unlabeled_dir.rglob("**/*.jpg"):
        photo_id = downloaded_photos.get(str(photo), {}).get("photo_id")
        if photo_id:
            insert_tags = [UNCERTAIN_TAG] if remaining_uncertain else None
            update_photo_tags(flickr_client, photo_id, insert_tags=insert_tags, remove_tags=[LABELED_TAG])
        else:
            typer.echo(f"Couldn't find a photo id for {photo}, please remove the tag manually.")

    shutil.rmtree(unlabeled_dir)


@app.command()
def remove_labeled_photos_from_flickr():
    raise NotImplementedError


@app.command()
def print_instructions():
    instructions = """
        Overview of the labeling process:

        1. User requests unlabeled photos from Flickr. These photos are downloaded to a specified
        "unlabeled" folder under the structure:

        correct/
            predicted_breed/
                hash.jpg
        wrong/
            predicted_breed/
        unsure/
            predicted_breed/

        2. User drags photos from the "unlabeled" folder into the "labeled" folder of the same root
        directory, conforming to the structure:

        labeled_breed/
            hash.jpg
        labeled_breed/
            hash.jpg

        4. User copies photos from "labeled" folder into the Google Drive dataset directory.

        5. We'll tag photos on Flickr as "labeled" as soon as they're downloaded to a
        user's machine in Step 1. If the user does not finish labeling all of the photos,
        they can remove the "labeled" tag from the remaining photos by running the command
        to return photos to the labeling pool. We'll do this for all of the photos remaining
        in the "unlabeled" folder.

        6. (Optional) Flickr limits our album to 1,000 photos. If needed we can delete photos
        which have already been marked as labeled. However, we should coordinate with all of
        the labelers and make sure they don't need to return any to the queue before doing
        this.
    """
    print(instructions)
