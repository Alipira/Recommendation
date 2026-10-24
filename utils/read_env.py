import os
from dotenv import find_dotenv, load_dotenv
from loguru import logger


def read_redis_env_vars():
    """
    Read the Redis settings.

    Values come from a .env file if one is found, otherwise from the normal
    environment variables. Missing values are returned as None, so the caller
    can say which one is missing.
    """
    env_vars_path = find_dotenv()

    if env_vars_path:
        try:
            load_dotenv(env_vars_path)
        except Exception as e:
            logger.error(f"Error loading env file: {e}")
    else:
        logger.warning(".env file not found, using environment variables only.")

    return (
        os.getenv("REDIS_HOST"),
        os.getenv("REDIS_PORT"),
        os.getenv("REDIS_PASSWORD"),
        os.getenv("STREAM_KEY"),
        os.getenv("STREAM_KEY_VAL"),
        os.getenv("STREAM_KEY_TEST"),
        os.getenv("GROUP_NAME"),
        os.getenv("CONSUMER_NAME"),
    )
