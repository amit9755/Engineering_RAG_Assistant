# ============================================================
# infra/terraform/main.tf - GCP Infrastructure as Code
#
# Learning Note:
#   Terraform lets you define cloud infrastructure in code.
#   Instead of clicking through the GCP console, you write
#   .tf files and run "terraform apply" to create everything.
#
#   Key benefits:
#   - Reproducible: same infra every time
#   - Version controlled: track changes in git
#   - Destroyable: "terraform destroy" tears it all down cleanly
#
#   Resources created here:
#     - Cloud Run service (hosts the API container)
#     - Cloud Storage bucket (document storage)
#     - Artifact Registry (Docker image storage)
#     - Secret Manager secrets (API keys)
#     - BigQuery dataset (eval metrics)
#     - IAM service account (security)
# ============================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  # Uncomment to store state in GCS (recommended for teams)
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "advanced-rag/state"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ===== VARIABLES =====
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Application name (used as prefix for resources)"
  type        = string
  default     = "advanced-rag"
}

variable "container_image" {
  description = "Docker image URL in Artifact Registry"
  type        = string
  default     = ""
}

# ===== ENABLE REQUIRED APIs =====
# Learning Note: GCP services must be explicitly enabled before use
resource "google_project_service" "services" {
  for_each = toset([
    "run.googleapis.com",           # Cloud Run
    "storage.googleapis.com",       # Cloud Storage
    "bigquery.googleapis.com",      # BigQuery
    "secretmanager.googleapis.com", # Secret Manager
    "artifactregistry.googleapis.com", # Container registry
    "aiplatform.googleapis.com",    # Vertex AI
  ])
  service            = each.key
  disable_on_destroy = false
}

# ===== SERVICE ACCOUNT =====
# Learning Note: Service accounts are non-human identities for apps.
# This follows the principle of least privilege - only grant what is needed.
resource "google_service_account" "rag_sa" {
  account_id   = "${var.app_name}-sa"
  display_name = "Advanced RAG Service Account"
  description  = "Identity used by the RAG application on Cloud Run"
}

# Grant necessary permissions to the service account
resource "google_project_iam_member" "rag_sa_roles" {
  for_each = toset([
    "roles/aiplatform.user",          # Use Vertex AI models
    "roles/storage.objectAdmin",      # Read/write GCS buckets
    "roles/bigquery.dataEditor",      # Write eval metrics to BigQuery
    "roles/secretmanager.secretAccessor", # Read API keys from Secret Manager
    "roles/logging.logWriter",        # Write structured logs
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.rag_sa.email}"
}

# ===== CLOUD STORAGE BUCKET =====
# Learning Note: GCS stores your source documents.
# A Pub/Sub notification can trigger ingestion when files arrive.
resource "google_storage_bucket" "documents" {
  name                        = "${var.project_id}-${var.app_name}-docs"
  location                    = var.region
  force_destroy               = false
  uniform_bucket_level_access = true

  lifecycle_rule {
    condition { age = 365 }
    action    { type = "Delete" }
  }

  versioning {
    enabled = true
  }
}

# ===== ARTIFACT REGISTRY =====
# Where Docker images are stored (like Docker Hub but private on GCP)
resource "google_artifact_registry_repository" "rag_repo" {
  location      = var.region
  repository_id = var.app_name
  description   = "Docker images for Advanced RAG System"
  format        = "DOCKER"

  depends_on = [google_project_service.services]
}

# ===== BIGQUERY DATASET =====
# Stores RAG evaluation metrics for dashboards and trend analysis
resource "google_bigquery_dataset" "rag_metrics" {
  dataset_id  = "rag_metrics"
  description = "RAG system evaluation metrics and query logs"
  location    = var.region

  delete_contents_on_destroy = false
}

resource "google_bigquery_table" "eval_metrics" {
  dataset_id = google_bigquery_dataset.rag_metrics.dataset_id
  table_id   = "eval_scores"
  description = "Per-query RAGAS evaluation scores"

  schema = jsonencode([
    { name = "timestamp",         type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "session_id",        type = "STRING",    mode = "NULLABLE" },
    { name = "query",             type = "STRING",    mode = "NULLABLE" },
    { name = "faithfulness",      type = "FLOAT",     mode = "NULLABLE" },
    { name = "answer_relevancy",  type = "FLOAT",     mode = "NULLABLE" },
    { name = "context_precision", type = "FLOAT",     mode = "NULLABLE" },
    { name = "overall_score",     type = "FLOAT",     mode = "NULLABLE" },
    { name = "latency_ms",        type = "INTEGER",   mode = "NULLABLE" },
    { name = "model_used",        type = "STRING",    mode = "NULLABLE" },
  ])
}

# ===== SECRET MANAGER =====
# Stores API keys securely (never hardcoded in code or env files)
resource "google_secret_manager_secret" "openai_key" {
  secret_id = "${var.app_name}-openai-api-key"
  replication {
    auto {}
  }
  depends_on = [google_project_service.services]
}

resource "google_secret_manager_secret" "langfuse_secret" {
  secret_id = "${var.app_name}-langfuse-secret-key"
  replication {
    auto {}
  }
}

# ===== CLOUD RUN SERVICE =====
# Learning Note: Cloud Run runs your Docker container serverlessly.
# It scales to zero when idle (no traffic = no cost) and scales
# up automatically under load. Perfect for RAG APIs.
resource "google_cloud_run_v2_service" "rag_api" {
  name     = var.app_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"  # allow public traffic

  template {
    service_account = google_service_account.rag_sa.email

    scaling {
      min_instance_count = 0   # scale to zero when idle
      max_instance_count = 10  # max 10 instances under load
    }

    containers {
      image = var.container_image != "" ? var.container_image : "gcr.io/cloudrun/hello"

      resources {
        limits = {
          cpu    = "2"      # 2 vCPUs
          memory = "4Gi"    # 4GB RAM (needed for embedding models)
        }
        cpu_idle          = true   # CPU only allocated during requests
        startup_cpu_boost = true   # Extra CPU during cold start
      }

      # Environment variables from Secret Manager
      env {
        name = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "OPENAI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.openai_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "LANGFUSE_SECRET_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.langfuse_secret.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "ENVIRONMENT"
        value = "production"
      }
      env {
        name  = "VECTOR_STORE_TYPE"
        value = "chroma"
      }
      env {
        name  = "GCP_BUCKET_NAME"
        value = google_storage_bucket.documents.name
      }

      # Health check
      startup_probe {
        http_get {
          path = "/api/v1/health/live"
        }
        initial_delay_seconds = 30
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 5
      }

      liveness_probe {
        http_get {
          path = "/api/v1/health/live"
        }
        period_seconds    = 30
        timeout_seconds   = 5
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_service_account.rag_sa,
    google_project_iam_member.rag_sa_roles,
  ]
}

# Allow unauthenticated requests (public API)
resource "google_cloud_run_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.rag_api.location
  service  = google_cloud_run_v2_service.rag_api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ===== OUTPUTS =====
output "cloud_run_url" {
  value       = google_cloud_run_v2_service.rag_api.uri
  description = "Public URL of the deployed RAG API"
}

output "gcs_bucket_name" {
  value       = google_storage_bucket.documents.name
  description = "GCS bucket for document uploads"
}

output "artifact_registry" {
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${var.app_name}"
  description = "Artifact Registry path for Docker images"
}

output "bigquery_dataset" {
  value       = google_bigquery_dataset.rag_metrics.dataset_id
  description = "BigQuery dataset for eval metrics"
}
