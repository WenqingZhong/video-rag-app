# The app's settings, in SSM Parameter Store under /video-rag/. deploy.sh writes them to the server's .env.
# Generated secrets live in the Terraform state too (keep it local and private). The third-party secrets (Pexels,
# Telegram, Anthropic) are NOT managed here, so they never touch the state or a file: put them in yourself (see infra/aws/README.md).

resource "random_password" "secret" {
  for_each = toset(["SESSION_SECRET", "SERVICE_TOKEN", "ADMIN_TOKEN", "POSTGRES_PASSWORD", "GRAFANA_ADMIN_PASSWORD"])
  length   = 48
  special  = false # used in URLs and shell
}

resource "aws_ssm_parameter" "secret" {
  for_each = random_password.secret
  name     = "/${var.project}/${each.key}"
  type     = "SecureString"
  value    = each.value.result
}

resource "aws_ssm_parameter" "setting" {
  for_each = {
    LLM_PROVIDER                = var.llm_provider
    SITE_ADDRESS                = local.site_address
    IMAGE_REGISTRY              = local.registry
    S3_BUCKET                   = aws_s3_bucket.app.bucket
    S3_REGION                   = var.region
    BEDROCK_REGION              = var.region
    BEDROCK_TEXT_MODEL_ID       = var.bedrock_text_model_id
    BEDROCK_VISION_MODEL_ID     = var.bedrock_vision_model_id
    LIMIT_GLOBAL_TOKENS_PER_DAY = tostring(var.global_tokens_per_day)
    UPLOAD_MAX_MB               = "100" # Caddy's request limit too
  }
  name  = "/${var.project}/${each.key}"
  type  = "String"
  value = each.value
}
