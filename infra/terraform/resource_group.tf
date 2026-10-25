# -----------------------------------------------------------------------------
# Resource Group: a single console view of every resource in this deployment.
#
# All resources are tagged (via provider default_tags) with Project + Environment.
# This group runs a tag query so the whole stack shows up together under
# "Resource Groups" in the AWS console, instead of hunting service-by-service.
# -----------------------------------------------------------------------------

resource "aws_resourcegroups_group" "all" {
  name        = "${local.name}-resources"
  description = "All ${var.project_name} ${var.environment} resources grouped by tag."

  resource_query {
    query = jsonencode({
      ResourceTypeFilters = ["AWS::AllSupported"]
      TagFilters = [
        {
          Key    = "Project"
          Values = ["taxi-forecasting"]
        },
        {
          Key    = "Environment"
          Values = [var.environment]
        },
      ]
    })
  }
}
