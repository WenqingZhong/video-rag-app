resource "aws_s3_bucket" "app" {
  bucket_prefix = "${var.project}-"
}

resource "aws_s3_bucket_public_access_block" "app" {
  bucket                  = aws_s3_bucket.app.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true # clips are shared through presigned URLs (1 hour), never public objects
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app" {
  bucket = aws_s3_bucket.app.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "app" {
  bucket = aws_s3_bucket.app.id
  rule {
    id     = "backups-14-days"
    status = "Enabled"
    filter {
      prefix = "backups/"
    }
    expiration {
      days = 14
    }
  }
  rule {
    id     = "chat-images-1-day" # photos sent for image search are needed for one request only
    status = "Enabled"
    filter {
      prefix = "chat-images/"
    }
    expiration {
      days = 1
    }
  }
  rule {
    id     = "deploy-bundles-30-days"
    status = "Enabled"
    filter {
      prefix = "deploy/"
    }
    expiration {
      days = 30
    }
  }
}

resource "aws_ecr_repository" "images" {
  for_each             = toset(["video-rag-app", "video-rag-embedder", "video-rag-ops"])
  name                 = each.key
  image_tag_mutability = "IMMUTABLE" # a tag (the commit) always means the same image
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "images" {
  for_each   = aws_ecr_repository.images
  repository = each.value.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}
