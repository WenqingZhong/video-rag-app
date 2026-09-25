#!/usr/bin/env bash
# Airflow keeps its own metadata DB ("airflow") on our Postgres server. Create it if missing, then start.
set -euo pipefail

python - <<'PY'
import os
import time

import psycopg2
from sqlalchemy.engine import make_url

url = make_url(os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"])
target = url.database
for attempt in range(30):
    try:
        conn = psycopg2.connect(host=url.host, port=url.port or 5432, user=url.username, password=url.password, dbname="postgres")
        break
    except psycopg2.OperationalError:
        time.sleep(2)
else:
    raise SystemExit("Postgres not reachable")
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (target,))
    if cur.fetchone() is None:
        cur.execute(f'CREATE DATABASE "{target}"')
        print(f"Created database {target}")
conn.close()
PY

# Local development: one process runs api-server, scheduler, dag-processor and triggerer.
# Production splits these into separate services (see README).
exec airflow standalone
