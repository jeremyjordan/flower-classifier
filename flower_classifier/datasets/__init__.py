import os


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
        "GIS_API_KEY",
        "GIS_PROJECT_CX",
    }
    if not required_keys.issubset(set(os.environ.keys())):
        missing_keys = required_keys.difference(set(os.environ.keys()))
        raise ValueError(f"Missing environment variables for: {missing_keys}")

    secrets = {}
    for k in required_keys:
        secrets[k] = os.environ[k]

    return secrets
