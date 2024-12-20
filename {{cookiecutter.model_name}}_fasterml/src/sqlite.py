import os
import sqlite3
from typing import Any

import base_class_data
import pandas as pd
import yaml
from classes import DatabaseConfig
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from yaml2pyclass import CodeGenerator
from yaml_class import Config


def get_data(controller: str, database_config: dict[str, Any]) -> pd.DataFrame:
    """Gets data from various sql type databases

    Args:
        controller (str): defines controller to use depending in database
        database_config (dict[str, Any]): object container of database configuration

    Returns:
        pd.DataFrame: Dataframe containing data
    """
    db_url = URL.create(
        controller,
        username=database_config.user,
        password=database_config.password,
        host=database_config.hostname,
        port=database_config.port,
        database=database_config.database,
    )
    eng = create_engine(
        db_url,
        client_encoding=database_config.encoding,
    )
    data = pd.read_sql(database_config.query, eng)
    return data


def dump_sqlite(database_config: dict[str, Any]) -> pd.Series:
    """Dumps data to sqlite depending on incoming database type and connection

    Args:
        database_config (dict[str, Any]): configuration of databse to connect to and retrieve data from

    Raises:
        TypeError: Raises error if database type is not supported
    """
    data: pd.DataFrame
    conn = sqlite3.connect("data.db")
    # , Maria, Sqlite, , , csv
    if (database_config.adapter == "mysql") or (
        database_config.adapter == "mariadb"
    ):  # MySQL/MariaDB
        data = get_data("mysql+pymysql", database_config)
    elif database_config.adapter == "mssql":  # MsSQL
        data = get_data("mssql+pyodbc", database_config)
    elif database_config.adapter == "postgresql":  # PostgreSQL
        data = get_data("postgresql+pg8000", database_config)
    elif database_config.adapter == "sqlite3":  # SQLite
        eng = create_engine(f"sqlite:///{database_config.path}")
        data = pd.read_sql(database_config.query, eng)
    elif database_config.adapter == "csv":  # CSV
        data = pd.read_csv(
            database_config.path,
            sep=database_config.sep,
            encoding=database_config.encoding,
        )
    else:
        raise TypeError("Type of database not supported")
    data.to_sql("train_data", conn, if_exists="replace")
    conn.close()
    return data.iloc[0].drop(database_config.target)


def populate_class(route: str) -> CodeGenerator:
    """Populates class from yaml to reuse in other areas

    Args:
        route (str): route of the yaml file.

    Returns:
        Config: Class containing all config from yaml
    """
    config = Config.from_yaml(route)
    return config


def populate_sqlite(configs: list[dict[str, Any]]):
    """populates sqlite with data

    Args:
        configs (list[dict[str, Any]]): list of database_configs
    """
    base_path = os.path.join(os.path.abspath(os.curdir), "schemas")
    for config in configs:
        db_Config = DatabaseConfig(config)
        dtype = dump_sqlite(db_Config)
        yml_path = os.path.join(base_path, f"{db_Config.name}.yaml")
        with open(yml_path, "w+") as ff:
            yaml.dump(dtype.to_dict(), ff)
            base_class_data.Config.from_yaml(yml_path)
