# -----------------------------------------------------------------------------
# API-only AWS deployment for the taxi-forecasting service.
#
# Topology: ECS Fargate service behind an ALB, pulling a SELF-CONTAINED image
# from Amazon ECR. The image bakes in the MLflow registry, models and demand
# history, so no RDS/S3 is required.
#
# This configuration is appliable. It is NOT applied automatically - see
# README.md and scripts/aws/. Running it incurs cost (Fargate + ALB).
# -----------------------------------------------------------------------------

data "aws_region" "current" {}

# Use the account's default VPC and its subnets to keep the example self-contained.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

locals {
  name = "${var.project_name}-${var.environment}"
}

# --- ECR: image repository ----------------------------------------------------
resource "aws_ecr_repository" "api" {
  name                 = local.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the most recent images to avoid unbounded storage cost.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 20 images so any recent release can be rolled back"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
        }
        action = { type = "expire" }
      }
    ]
  })
}

# --- Security groups ----------------------------------------------------------
resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Allow inbound HTTP to the load balancer"
  vpc_id      = data.aws_vpc.default.id

  # With HTTPS on, only CloudFront may reach the load balancer (see cdn.tf).
  ingress {
    description     = var.enable_https ? "HTTP from CloudFront only" : "HTTP"
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    cidr_blocks     = var.enable_https ? [] : ["0.0.0.0/0"]
    prefix_list_ids = var.enable_https ? [data.aws_ec2_managed_prefix_list.cloudfront.id] : []
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "Allow traffic from the ALB to the ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "From ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- Application Load Balancer ------------------------------------------------
resource "aws_lb" "api" {
  name               = "${local.name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default.ids

  # Recursive multi-hour / all-zone demand forecasts can take a while; keep the
  # connection open well past the 60s default so slow /forecast/demand calls
  # don't return 504 to the dashboard.
  idle_timeout = 300
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# --- IAM: task execution role -------------------------------------------------
data "aws_iam_policy_document" "ecs_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "task_execution" {
  name               = "${local.name}-task-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume.json
}

# Grants ECR pull + CloudWatch Logs write for the Fargate agent.
resource "aws_iam_role_policy_attachment" "task_execution" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- ECS: cluster, task definition, service ----------------------------------
resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]
      # The image is self-contained; MLFLOW_TRACKING_URI is baked into the
      # Dockerfile. No RDS/S3 configuration is required. Access controls are
      # configured here: rate limit and CORS as plain settings, and the API key
      # as a secret read from SSM Parameter Store at task start.
      environment = [
        { name = "RATE_LIMIT_PER_MINUTE", value = tostring(var.api_rate_limit_per_minute) },
        # CloudFront and the load balancer each append to X-Forwarded-For.
        { name = "TRUSTED_PROXY_HOPS", value = var.enable_https ? "2" : "1" },
        { name = "CORS_ALLOW_ORIGINS", value = join(",", var.api_cors_allow_origins) },
      ]
      secrets = var.api_key == "" ? [] : [
        { name = "API_KEY", valueFrom = aws_ssm_parameter.api_key[0].arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "api" {
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.http]
}

# --- API key (optional): encrypted in SSM, injected as a container secret ------
resource "aws_ssm_parameter" "api_key" {
  count = var.api_key == "" ? 0 : 1
  name  = "/${local.name}/api-key"
  type  = "SecureString"
  value = var.api_key
}

resource "aws_iam_role_policy" "task_execution_api_key" {
  count = var.api_key == "" ? 0 : 1
  name  = "${local.name}-read-api-key"
  role  = aws_iam_role.task_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ssm:GetParameters"]
        Resource = [aws_ssm_parameter.api_key[0].arn]
      }
    ]
  })
}

