# Scheduled jobs in production: calls the API's admin endpoints and backs up Postgres to S3.
FROM postgres:16-alpine
RUN apk add --no-cache curl aws-cli
COPY crontab /etc/crontabs/root
COPY backup.sh entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/backup.sh /usr/local/bin/entrypoint.sh
# crond in the foreground; its jobs log to the container's output (docker compose logs ops)
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
