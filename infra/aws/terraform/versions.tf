terraform {
  required_version = ">= 1.6"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "~> 6.0" }
    random = { source = "hashicorp/random", version = "~> 3.6" }
  }
  # State is kept locally (terraform.tfstate, gitignored): it holds the generated app secrets, so don't commit it.
  # For a team, move it to an S3 backend with encryption and locking.
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = var.project, ManagedBy = "terraform" }
  }
}
