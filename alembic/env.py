from __future__ import annotations

from logging.config import fileConfig
from dotenv import load_dotenv

import sqlalchemy as sa
from sqlalchemy import engine_from_config, pool
from alembic import context

from football_video_analyser.db.models import Base
from football_video_analyser.db.config import get_db_schema, get_db_url

load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
config.set_main_option("sqlalchemy.url", get_db_url())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

SCHEMA = get_db_schema()
TARGET_METADATA = Base.metadata


def include_object(object_, name, type_, reflected, compare_to):
    if type_ == "schema":
        return name == SCHEMA
    return True


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=TARGET_METADATA,
        literal_binds=True,
        include_schemas=True,
        version_table_schema=SCHEMA if SCHEMA else None,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=TARGET_METADATA,
            include_schemas=True,
            version_table_schema=SCHEMA if SCHEMA else None,
            include_object=include_object,
        )

        with context.begin_transaction():
            if SCHEMA:
                connection.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"'))
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
