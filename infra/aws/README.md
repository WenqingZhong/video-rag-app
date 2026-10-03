# Deploying to AWS

One ARM EC2 instance running `compose.prod.yml`, with Claude Haiku 4.5 through the Anthropic API and files in
Amazon S3.

```text
GitHub (push to main) ── tests ── ARM images → ECR ── SSM Run Command ──▶ EC2 t4g.large (Docker Compose)
                                                                           caddy :443 ─▶ api, bot, workers, …
browser / Telegram ──HTTPS──▶ caddy        IAM role ──▶ S3 (videos, clips, backups)   API key ──▶ Anthropic API
```

**Cost:** about **$57/month** while it runs (instance $49, disk $3, public IP $4, ECR and S3 ~$1–2), plus model
usage on the Anthropic API, capped by the global daily token budget (500k tokens ≈ $0.70/day at most) and by the
spend limit you set in the Anthropic console.

## 1. Tools and an AWS login (once)

```bash
brew install awscli
brew install hashicorp/tap/terraform           # Terraform is no longer in Homebrew's main catalogue
brew install --cask session-manager-plugin   # admin access through SSM tunnels (step 8)
aws configure sso        # recommended (IAM Identity Center); or `aws configure` with an IAM user's access key
aws sts get-caller-identity                  # shows your account: you're signed in
```

Terraform needs an identity that can create IAM roles, EC2, S3, ECR and SSM parameters (e.g. AdministratorAccess).
Never paste keys into a chat or commit them.

## 2. An Anthropic API key (once)

At [console.anthropic.com](https://console.anthropic.com): add a few dollars of credit, set a monthly **spend limit**
(Settings → Limits), and create an API key. Put it in the project's `.env` (gitignored), for the local evaluation:

```text
ANTHROPIC_API_KEY=<the key>
```

(Amazon Bedrock also works, `LLM_PROVIDER=bedrock`, if the account has Bedrock model access.)

## 3. Evaluate Claude before switching (a few cents)

With the local stack running:

```bash
make eval-claude     # intents, chat routing (3 sets), QA, whole conversations → eval/results/claude/
```

Compare with `eval/results/*.json` (qwen2.5vl:3b) before deploying.

## 4. Create the infrastructure

```bash
cd infra/aws/terraform
terraform init
terraform plan -var alert_email=you@example.com      # read what will be created
terraform apply -var alert_email=you@example.com     # starts billing
terraform output                                     # site URL, instance id, bucket, GitHub variables
```

`terraform.tfstate` holds the generated app secrets: it stays on your Mac (gitignored).

## 5. The third-party secrets (typed, never in a file or shell history)

```bash
read -rs PEXELS && aws ssm put-parameter --name /video-rag/PEXELS_API_KEY --type SecureString --value "$PEXELS" && unset PEXELS
read -rs TELEGRAM && aws ssm put-parameter --name /video-rag/TELEGRAM_BOT_TOKEN --type SecureString --value "$TELEGRAM" && unset TELEGRAM
read -rs ANTHROPIC && aws ssm put-parameter --name /video-rag/ANTHROPIC_API_KEY --type SecureString --value "$ANTHROPIC" && unset ANTHROPIC
```

Or in the AWS console: **Systems Manager → Parameter Store → Create parameter**, type *SecureString*.

## 6. GitHub Actions

In the repository: **Settings → Secrets and variables → Actions → Variables**, add each entry of
`terraform output github_variables` (AWS_REGION, AWS_DEPLOY_ROLE, INSTANCE_ID, BUCKET, REGISTRY). None is secret: the
deploy role can only be assumed by this repository's `main` branch.

The image build uses GitHub's ARM runners (`ubuntu-24.04-arm`), free for public repositories.

## 7. First deploy

A Telegram bot token can be polled from one place only: **stop the local bot first**, or it and the server's will
take turns failing (`409 Conflict`).

```bash
docker compose stop telegram-bot     # local; for local bot work later, create a second bot with @BotFather
git push                             # to main: Actions → Deploy (≈ 10 min the first time: the images are built)
```

Then copy the library (35 videos, their keyframes and vectors; nothing is re-processed):

```bash
uv run python scripts/migrate_library.py --bucket <bucket> --instance-id <instance id> --dry-run
uv run python scripts/migrate_library.py --bucket <bucket> --instance-id <instance id>
```

Open the `site_url` output. The first HTTPS request may take a few seconds while the certificate is issued.

## 8. Operating it

| Task | How |
|---|---|
| Shell on the server | `aws ssm start-session --target <instance id>` then `cd /opt/video-rag && sudo docker compose -f compose.prod.yml ps` |
| Logs | `sudo docker compose -f compose.prod.yml logs -f --tail 100 api` |
| Grafana | `aws ssm start-session --target <id> --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["3000"],"localPortNumber":["3000"]}'`, open http://localhost:3000 (user `admin`, password: `aws ssm get-parameter --name /video-rag/GRAFANA_ADMIN_PASSWORD --with-decryption --query Parameter.Value --output text`) |
| API docs, admin endpoints | the same tunnel with port 8000; admin calls need `x-admin-token` (`/video-rag/ADMIN_TOKEN`) |
| Change a setting | `aws ssm put-parameter --name /video-rag/<NAME> --value … --overwrite`, then re-run the Deploy workflow |
| Restore the database | `aws s3 ls s3://<bucket>/backups/`, then `aws s3 cp s3://<bucket>/backups/<file> - \| gunzip \| sudo docker compose -f compose.prod.yml exec -T postgres psql -U video_rag video_rag` |
| Pause (no instance cost) | `aws ec2 stop-instances --instance-ids <id>`; start again with `start-instances` (the IP stays) |
| Remove everything | empty the bucket, then `terraform destroy` |
