"""
Manage PostgreSQL and Redis connections for the ERP cache simulation.

This module maps the storage layer from Paper Section III.A and SPEC.md
Section 5 to local PostgreSQL and Redis services.
"""

import sys

import psycopg2
import redis

from config import DUMMY_ROWS_PER_TABLE, ENTITY_TABLE_NAMES


def get_postgres_conn() -> psycopg2.extensions.connection:
    """Connect to PostgreSQL with the hardcoded local experiment credentials.

    Returns:
        psycopg2.extensions.connection: Open PostgreSQL connection.
    """
    try:
        return psycopg2.connect(
            host="localhost",
            port=5433,
            dbname="erp_cache",
            user="postgres",
            password="postgres",
        )
    except psycopg2.Error as exc:
        print(f"[DB] PostgreSQL connection failed: {exc}")
        sys.exit(1)


def get_redis_conn() -> redis.Redis:
    """Connect to Redis and disable automatic Redis eviction.

    Returns:
        redis.Redis: Open Redis connection with maxmemory-policy noeviction.
    """
    try:
        redis_conn = redis.Redis(host="localhost", port=6379, decode_responses=True)
        redis_conn.ping()
        redis_conn.config_set("maxmemory-policy", "noeviction")
        return redis_conn
    except redis.RedisError as exc:
        print(f"[DB] Redis connection failed: {exc}")
        sys.exit(1)


def setup_postgres_tables(conn: psycopg2.extensions.connection) -> None:
    """Create ERP tables and insert 100 dummy rows per table.

    Args:
        conn: Open PostgreSQL connection.
    """
    table_names = ENTITY_TABLE_NAMES

    try:
        with conn.cursor() as cursor:
            for table_name in table_names:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        data TEXT
                    )
                    """)

            for table_name in table_names:
                cursor.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY")
                rows = [
                    (f"dummy_{table_name}_{i}",)
                    for i in range(1, DUMMY_ROWS_PER_TABLE + 1)
                ]
                cursor.executemany(
                    f"INSERT INTO {table_name} (data) VALUES (%s)",
                    rows,
                )

        conn.commit()
    except psycopg2.Error as exc:
        try:
            conn.rollback()
        except psycopg2.Error as rollback_exc:
            print(f"[DB] PostgreSQL rollback failed: {rollback_exc}")
            sys.exit(1)
        print(f"[DB] PostgreSQL setup failed: {exc}")
        sys.exit(1)
