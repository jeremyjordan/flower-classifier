import os
import sqlite3
import webbrowser
from pathlib import Path

OAUTH_TOKEN_CACHE = Path.home() / ".flickr" / "oauth-tokens.sqlite"


def authenticate_new_user(flickr_client):
    """
    Example code snippet to authenticate for a new user.
    """
    flickr_client.get_request_token(oauth_callback="oob")
    authorize_url = flickr_client.auth_url(perms="write")
    webbrowser.open_new_tab(authorize_url)
    verifier = str(input("Verifier code: "))
    flickr_client.get_access_token(verifier)


def show_user_secrets(username="jeremythomasjordan"):
    conn = sqlite3.connect(OAUTH_TOKEN_CACHE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    results = cur.execute(
        f"""
        SELECT *
        FROM oauth_tokens
        WHERE username = '{username}';
        """
    )
    results = dict(results.fetchone())
    user_secrets = {
        "OAUTH_TOKEN": results["oauth_token"],
        "OAUTH_TOKEN_SECRET": results["oauth_token_secret"],
        "OAUTH_ACCESS_LEVEL": results["access_level"],
        "OAUTH_FULL_NAME": results["fullname"],
        "OAUTH_USERNAME": results["username"],
        "OAUTH_USER_NSID": results["user_nsid"],
    }
    return user_secrets


def get_secrets():
    required_keys = {
        "API_KEY",
        "API_SECRET",
        "OAUTH_TOKEN",
        "OAUTH_TOKEN_SECRET",
        "OAUTH_ACCESS_LEVEL",
        "OAUTH_FULL_NAME",
        "OAUTH_USERNAME",
        "OAUTH_USER_NSID",
    }
    if not required_keys.issubset(set(os.environ.keys())):
        missing_keys = required_keys.difference(set(os.environ.keys()))
        raise ValueError(f"Missing environment variables for: {missing_keys}")

    secrets = {}
    for k in required_keys:
        secrets[k] = os.environ[k]

    return secrets
