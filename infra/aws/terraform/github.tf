# GitHub Actions deploys with short-lived credentials (OIDC): no AWS keys are stored in GitHub.
# Only pushes to main of var.github_repo can assume this role.
resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

resource "aws_iam_role" "deploy" {
  name = "${var.project}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          # main only (a list means any of them). Current GitHub tokens name the repository by its permanent ids;
          # older ones by name
          "token.actions.githubusercontent.com:sub" = [
            "repo:${var.github_repo_ids}:ref:refs/heads/main",
            "repo:${var.github_repo}:ref:refs/heads/main",
          ]
        }
      }
    }]
  })
}

resource "aws_iam_role_policy" "deploy" {
  name = "deploy"
  role = aws_iam_role.deploy.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Sid = "EcrLogin", Effect = "Allow", Action = "ecr:GetAuthorizationToken", Resource = "*" },
      {
        Sid    = "EcrPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage", "ecr:DescribeImages",
        ]
        Resource = [for r in aws_ecr_repository.images : r.arn]
      },
      {
        Sid      = "Bundle"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.app.arn}/deploy/*"
      },
      {
        Sid    = "RunDeploy"
        Effect = "Allow"
        Action = "ssm:SendCommand"
        Resource = [
          aws_instance.server.arn,
          "arn:aws:ssm:${var.region}::document/AWS-RunShellScript",
        ]
      },
      { Sid = "DeployResult", Effect = "Allow", Action = ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"], Resource = "*" },
    ]
  })
}
