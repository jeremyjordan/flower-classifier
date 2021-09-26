import sqlite3
import webbrowser
from pathlib import Path

OAUTH_TOKEN_CACHE = Path.home() / ".flickr" / "oauth-tokens.sqlite"
USER_ID = "192840150@N08"


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
