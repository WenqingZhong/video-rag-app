output "site_url" {
  value = "https://${local.site_address}/app"
}

output "instance_id" {
  value = aws_instance.server.id
}

output "public_ip" {
  value = aws_eip.server.public_ip
}

output "bucket" {
  value = aws_s3_bucket.app.bucket
}

output "registry" {
  value = local.registry
}

# Put these in the GitHub repository: Settings → Secrets and variables → Actions → Variables (none are secret)
output "github_variables" {
  value = {
    AWS_REGION      = var.region
    AWS_DEPLOY_ROLE = aws_iam_role.deploy.arn
    INSTANCE_ID     = aws_instance.server.id
    BUCKET          = aws_s3_bucket.app.bucket
    REGISTRY        = local.registry
  }
}
