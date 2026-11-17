# --- HTTPS through CloudFront --------------------------------------------------
# CloudFront terminates TLS with its default *.cloudfront.net certificate, so the
# API key and every request travel encrypted without a custom domain. Nothing is
# cached: every request is forwarded to the load balancer with all its headers,
# including X-API-Key and the Host header Streamlit checks on websockets.
# The load balancers accept traffic only from CloudFront's origin-facing
# addresses (see the security groups), so the plain-HTTP path is closed to the
# public and the client IP that CloudFront appends to X-Forwarded-For can be
# trusted (TRUSTED_PROXY_HOPS = 2 on the API). Those addresses are shared by all
# of CloudFront, so our distributions also send a secret header that the
# listeners require: a distribution someone else points at the load balancer
# gets 403.

locals {
  origin_verify_header = "X-Origin-Verify"
}

resource "random_password" "origin_verify" {
  count   = var.enable_https ? 1 : 0
  length  = 40
  special = false
}

data "aws_cloudfront_cache_policy" "disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewer"
}

data "aws_ec2_managed_prefix_list" "cloudfront" {
  name = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_cloudfront_distribution" "api" {
  count        = var.enable_https ? 1 : 0
  enabled      = true
  comment      = "${local.name} API over HTTPS"
  price_class  = "PriceClass_100"
  http_version = "http2and3"

  origin {
    origin_id   = "api-alb"
    domain_name = aws_lb.api.dns_name

    custom_header {
      name  = local.origin_verify_header
      value = random_password.origin_verify[0].result
    }

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 60
    }
  }

  default_cache_behavior {
    target_origin_id         = "api-alb"
    viewer_protocol_policy   = "https-only"
    allowed_methods          = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_cloudfront_distribution" "dashboard" {
  count        = var.enable_https && var.enable_dashboard ? 1 : 0
  enabled      = true
  comment      = "${local.name} dashboard over HTTPS"
  price_class  = "PriceClass_100"
  http_version = "http2and3"

  origin {
    origin_id   = "dashboard-alb"
    domain_name = aws_lb.dashboard[0].dns_name

    custom_header {
      name  = local.origin_verify_header
      value = random_password.origin_verify[0].result
    }

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 60
    }
  }

  default_cache_behavior {
    target_origin_id         = "dashboard-alb"
    viewer_protocol_policy   = "redirect-to-https"
    allowed_methods          = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods           = ["GET", "HEAD"]
    cache_policy_id          = data.aws_cloudfront_cache_policy.disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
    compress                 = true
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

# Forward only requests that carry the secret header our distributions add.
resource "aws_lb_listener_rule" "api_from_cloudfront" {
  count        = var.enable_https ? 1 : 0
  listener_arn = aws_lb_listener.http.arn
  priority     = 1

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }

  condition {
    http_header {
      http_header_name = local.origin_verify_header
      values           = [random_password.origin_verify[0].result]
    }
  }
}

resource "aws_lb_listener_rule" "dashboard_from_cloudfront" {
  count        = var.enable_https && var.enable_dashboard ? 1 : 0
  listener_arn = aws_lb_listener.dashboard_http[0].arn
  priority     = 1

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dashboard[0].arn
  }

  condition {
    http_header {
      http_header_name = local.origin_verify_header
      values           = [random_password.origin_verify[0].result]
    }
  }
}
