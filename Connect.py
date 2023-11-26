import pandas as pd

from configparser import ConfigParser
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from loguru import logger


config_pareser = ConfigParser()
config_pareser.read('../config.ini')

username = config_pareser['server']['user_07']
password = config_pareser['server']['pass_07']
hostip = config_pareser['server']['host_07']
dabase_name = config_pareser['server']['database_name']


# define a function to connect database
def connect_to_db(query, date_column=None):
    """Connecting to Datawarehouse and read the tables by pandas library

    Returns:
        Pandas dataframe
    """
    try:
        url_object = URL.create(
            "mssql+pyodbc",
            username=username,
            password=password,
            host=hostip,
            database=dabase_name,
            query=dict(driver='ODBC Driver 17 for SQL Server')
        )
        engine = create_engine(url_object)
    except Exception as e:
        print(f'Error is: {e}')

    logger.info('Query database...')
    df = pd.read_sql_query(text(query), con=engine.connect())
    logger.info('Query is done!')
    return df
