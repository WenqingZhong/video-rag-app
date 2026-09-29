data "aws_caller_identity" "me" {}

data "aws_ssm_parameter" "al2023_arm" {
  name = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64"
}

# ---- what the server may do (its IAM role; no keys on the machine) ----------------------------------------------
resource "aws_iam_role" "server" {
  name = "${var.project}-server"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "ssm_core" { # Session Manager and Run Command (deploys)
  role       = aws_iam_role.server.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.server.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy" "server" {
  name = "app"
  role = aws_iam_role.server.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "Bucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.app.arn
      },
      {
        Sid      = "Objects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
        Resource = "${aws_s3_bucket.app.arn}/*"
      },
      {
        Sid    = "Bedrock"
        Effect = "Allow"
        Action = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        # The cross-region profile, and the model in each US region it may route to
        Resource = [
          "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.me.account_id}:inference-profile/${var.bedrock_text_model_id}",
          "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.me.account_id}:inference-profile/${var.bedrock_vision_model_id}",
          "arn:aws:bedrock:*::foundation-model/anthropic.claude-haiku-4-5-*",
        ]
      },
      {
        Sid      = "Settings"
        Effect   = "Allow"
        Action   = ["ssm:GetParametersByPath", "ssm:GetParameters", "ssm:GetParameter"]
        Resource = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.me.account_id}:parameter/${var.project}/*"
      },
    ]
  })
}

resource "aws_iam_instance_profile" "server" {
  name = "${var.project}-server"
  role = aws_iam_role.server.name
}

# ---- the server ----------------------------------------------------------------------------------------------------
resource "aws_instance" "server" {
  ami                    = data.aws_ssm_parameter.al2023_arm.value
  instance_type          = var.instance_type
  subnet_id              = data.aws_subnets.default.ids[0]
  vpc_security_group_ids = [aws_security_group.web.id]
  iam_instance_profile   = aws_iam_instance_profile.server.name
  user_data              = file("${path.module}/../bootstrap.sh")

  root_block_device {
    volume_size = var.disk_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens                 = "required" # IMDSv2 only
    http_put_response_hop_limit = 2          # containers are one hop further: they need it for the IAM role
  }

  lifecycle {
    ignore_changes = [ami, user_data] # a newer AMI doesn't replace the running server
  }

  tags = { Name = var.project }
}

resource "aws_eip" "server" {
  instance = aws_instance.server.id
  domain   = "vpc"
}

locals {
  site_address = var.site_address != "" ? var.site_address : "${replace(aws_eip.server.public_ip, ".", "-")}.sslip.io"
  registry     = "${data.aws_caller_identity.me.account_id}.dkr.ecr.${var.region}.amazonaws.com"
}
