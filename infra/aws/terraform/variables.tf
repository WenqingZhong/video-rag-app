variable "project" {
  type    = string
  default = "video-rag"
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "instance_type" {
  description = "8 GB, ARM (Graviton): fits the stack without a local model; images build natively on an M-series Mac"
  type        = string
  default     = "t4g.large"
}

variable "disk_gb" {
  type    = number
  default = 40
}

variable "github_repo" {
  description = "owner/name: the only repository whose main branch may deploy"
  type        = string
  default     = "WenqingZhong/video-rag-app"
}

variable "site_address" {
  description = "Hostname for HTTPS. Empty: a free <ip>.sslip.io name that resolves to the server's Elastic IP"
  type        = string
  default     = ""
}

variable "llm_provider" {
  description = "anthropic (the Anthropic API; key put in SSM by hand) or bedrock (needs Bedrock model access)"
  type        = string
  default     = "anthropic"
}

variable "bedrock_text_model_id" {
  description = "Claude Haiku 4.5 through the US cross-region inference profile"
  type        = string
  default     = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "bedrock_vision_model_id" {
  type    = string
  default = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "global_tokens_per_day" {
  description = "Ceiling on the whole app's model tokens per day (≈ $0.70 at Haiku 4.5 prices for 500k)"
  type        = number
  default     = 500000
}

variable "alert_email" {
  description = "Where AWS emails budget alerts (empty: no budget alarm)"
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  type    = number
  default = 80
}
