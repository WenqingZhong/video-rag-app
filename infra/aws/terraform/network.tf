# The default VPC: one public instance needs no custom networking.
data "aws_vpc" "default" {
  default = true
}

# Not every zone offers every instance type (us-east-1e has no t4g): only subnets in zones that offer ours.
data "aws_ec2_instance_type_offerings" "here" {
  location_type = "availability-zone"
  filter {
    name   = "instance-type"
    values = [var.instance_type]
  }
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
  filter {
    name   = "availability-zone"
    values = data.aws_ec2_instance_type_offerings.here.locations
  }
}

# Only HTTP(S) in. No SSH: administration goes through SSM Session Manager (IAM-authenticated, logged).
resource "aws_security_group" "web" {
  name        = "${var.project}-web"
  description = "HTTPS to Caddy; nothing else"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description      = "HTTP (redirects to HTTPS, certificate challenges)"
    from_port        = 80
    to_port          = 80
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  ingress {
    description      = "HTTPS"
    from_port        = 443
    to_port          = 443
    protocol         = "tcp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  ingress {
    description      = "HTTP/3"
    from_port        = 443
    to_port          = 443
    protocol         = "udp"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}
