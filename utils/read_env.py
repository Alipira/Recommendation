import os
from dotenv import find_dotenv, load_dotenv
from loguru import logger


def read_redis_env_vars():
    """Read Redis Config"""

    env_vars_path = find_dotenv()

    try:
        if env_vars_path:
            load_dotenv(env_vars_path)
            redis_host = os.getenv("REDIS_HOST")
            redis_port = os.getenv("REDIS_PORT")
            redis_password = os.getenv("REDIS_PASSWORD")
            stream_key = os.getenv("STREAM_KEY")
            stream_key_val = os.getenv("STREAM_KEY_VAL")
            stream_key_test = os.getenv("STREAM_KEY_TEST")
            group_name = os.getenv("GROUP_NAME")
            consumer_name = os.getenv("CONSUMER_NAME")

            return redis_host, redis_port, redis_password, stream_key, stream_key_val, stream_key_test, group_name, consumer_name
        else:
            logger.warning(".env file not found, skipping...")

    except Exception as e:
        logger.error(f"Error loading env file: {e}")