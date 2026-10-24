"""
ERP (SQL Server) adapter.

Fill in `ErpConfig` with your own queries. Each query must return the column
names shown in the comments; rename in SQL with AS. The defaults are only
examples based on the old feature names (PersonID, Group4Code, ...).

Customer vectors from CustomerVAE can be passed as a parquet file instead of
(or next to) the raw customer table. See load_user_vectors().

Needs: pip install sqlalchemy pyodbc
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hybridrec.data.schema import Dataset, normalise_interactions


@dataclass
class ErpConfig:
    # e.g. "mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+18+for+SQL+Server"
    # Read it from an environment variable; never commit it.
    connection_url: str

    # Must return: user_id, item_id, basket_id, order_date, qty
    orders_sql: str = """
        SELECT  h.PersonID      AS user_id,
                d.ProductID     AS item_id,
                h.InvoiceID     AS basket_id,
                h.InvoiceDate   AS order_date,
                d.Qty           AS qty
        FROM    SalesInvoiceHeader h
        JOIN    SalesInvoiceDetail d ON d.InvoiceID = h.InvoiceID
        WHERE   h.InvoiceDate >= :start_date
    """

    # Must return: item_id, title, and the columns in item_text_cols
    items_sql: str = """
        SELECT  ProductID   AS item_id,
                ProductName AS title,
                Group1Name  AS group1,
                Group2Name  AS group2,
                FlavorName  AS flavor
        FROM    Product
    """
    item_text_cols: tuple[str, ...] = ("group1", "group2", "flavor")

    # Optional. Must return user_id plus numeric columns only.
    users_sql: str | None = None

    start_date: str = "2023-01-01"


def load_erp(cfg: ErpConfig) -> Dataset:
    from sqlalchemy import create_engine, text  # imported here: optional dependency

    engine = create_engine(cfg.connection_url)
    with engine.connect() as conn:
        orders = pd.read_sql(text(cfg.orders_sql), conn, params={"start_date": cfg.start_date})
        items = pd.read_sql(text(cfg.items_sql), conn)
        users = pd.read_sql(text(cfg.users_sql), conn) if cfg.users_sql else None

    origin = pd.Timestamp(cfg.start_date)
    orders["ts"] = (pd.to_datetime(orders["order_date"]) - origin).dt.total_seconds() / 86400.0
    interactions = normalise_interactions(orders)

    items["item_id"] = items["item_id"].astype(str)
    items = items.drop_duplicates("item_id")
    if users is not None:
        users["user_id"] = users["user_id"].astype(str)

    return Dataset(
        interactions=interactions,
        items=items,
        users=users,
        item_text_cols=list(cfg.item_text_cols),
        name="erp",
    ).validate()


def load_user_vectors(path: str | Path) -> pd.DataFrame:
    """
    Load precomputed customer vectors (e.g. CustomerVAE latent means).

    Expected parquet layout: a `user_id` column plus one float column per
    dimension. The result can be passed as Dataset.users or straight to the
    graph model as user node features.
    """
    vecs = pd.read_parquet(path)
    if "user_id" not in vecs.columns:
        raise ValueError(f"{path} has no user_id column")
    vecs["user_id"] = vecs["user_id"].astype(str)
    return vecs
