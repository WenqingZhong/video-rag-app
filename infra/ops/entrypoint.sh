#!/bin/sh
# crond starts jobs with an empty environment: save the few values they need where they can read them.
set -eu
: > /etc/ops.env
for name in ADMIN_TOKEN PGPASSWORD BACKUP_BUCKET; do
  eval "value=\${$name}"
  printf "export %s='%s'\n" "$name" "$value" >> /etc/ops.env
done
chmod 600 /etc/ops.env
exec crond -f -l 8
