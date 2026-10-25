# -----------------------------------------------------------------------------
# Streamlit showcase dashboard (opt-in).
#
# A second ECS Fargate service behind its own ALB, serving the Streamlit
# dashboard image from a dedicated ECR repo. The dashboard is stateless and
# talks to the API service over HTTP via API_BASE_URL (the API ALB DNS).
#
# Everything here is gated by `enable_dashboard` (default false) so it incurs
# no cost unless explicitly turned on, mirroring the CI/CD stack pattern.
# -----------------------------------------------------------------------------

locals {
  dashboard_name = "${var.project_name}-${var.environment}-dashboard"
  # ALB and target-group names are capped at 32 chars, so use a compact variant.
  dashboard_lb_name = "${replace(var.project_name, "forecasting", "fc")}-${var.environment}-dash"
}

# --- ECR: dashboard image repository -----------------------------------------
resource "aws_ecr_repository" "dashboard" {
  count                = var.enable_dashboard ? 1 : 0
  name                 = local.dashboard_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "dashboard" {
  count      = var.enable_dashboard ? 1 : 0
  repository = aws_ecr_repository.dashboard[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = { type = "expire" }
      }
    ]
  })
}

# --- Security groups ----------------------------------------------------------
resource "aws_security_group" "dashboard_alb" {
  count       = var.enable_dashboard ? 1 : 0
  name        = "${local.dashboard_name}-alb"
  description = "Allow inbound HTTP to the dashboard load balancer"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "dashboard_service" {
  count       = var.enable_dashboard ? 1 : 0
  name        = "${local.dashboard_name}-service"
  description = "Allow traffic from the dashboard ALB to the ECS tasks"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "From dashboard ALB"
    from_port       = var.dashboard_container_port
    to_port         = var.dashboard_container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.dashboard_alb[0].id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- Application Load Balancer ------------------------------------------------
resource "aws_lb" "dashboard" {
  count              = var.enable_dashboard ? 1 : 0
  name               = "${local.dashboard_lb_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.dashboard_alb[0].id]
  subnets            = data.aws_subnets.default.ids

  # Streamlit runs the page script synchronously; while it waits on a slow
  # forecast call the browser connection is idle, so keep it open past 60s to
  # avoid the "Connection error / 504" toast.
  idle_timeout = 300
}

resource "aws_lb_target_group" "dashboard" {
  count       = var.enable_dashboard ? 1 : 0
  name        = "${local.dashboard_lb_name}-tg"
  port        = var.dashboard_container_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  # Streamlit exposes a lightweight health endpoint.
  health_check {
    path                = "/_stcore/health"
    matcher             = "200"
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "dashboard_http" {
  count             = var.enable_dashboard ? 1 : 0
  load_balancer_arn = aws_lb.dashboard[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.dashboard[0].arn
  }
}

# --- ECS: task definition + service ------------------------------------------
resource "aws_cloudwatch_log_group" "dashboard" {
  count             = var.enable_dashboard ? 1 : 0
  name              = "/ecs/${local.dashboard_name}"
  retention_in_days = 14
}

resource "aws_ecs_task_definition" "dashboard" {
  count                    = var.enable_dashboard ? 1 : 0
  family                   = local.dashboard_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.dashboard_task_cpu
  memory                   = var.dashboard_task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn

  container_definitions = jsonencode([
    {
      name      = "dashboard"
      image     = "${aws_ecr_repository.dashboard[0].repository_url}:${var.dashboard_image_tag}"
      essential = true
      portMappings = [
        {
          containerPort = var.dashboard_container_port
          protocol      = "tcp"
        }
      ]
      # Point the dashboard at the API service's public ALB.
      environment = [
        {
          name  = "API_BASE_URL"
          value = "http://${aws_lb.api.dns_name}"
        },
        {
          name  = "PYTHONFAULTHANDLER"
          value = "1"
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.dashboard[0].name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "dashboard"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "dashboard" {
  count           = var.enable_dashboard ? 1 : 0
  name            = local.dashboard_name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.dashboard[0].arn
  desired_count   = var.dashboard_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.dashboard_service[0].id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.dashboard[0].arn
    container_name   = "dashboard"
    container_port   = var.dashboard_container_port
  }

  depends_on = [aws_lb_listener.dashboard_http]
}
