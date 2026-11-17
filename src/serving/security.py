"""Access controls for the prediction API: API key, rate limit and CORS.

Each control is configured by an environment variable so the same image runs
locally (all off) and in production (set by the ECS task definition):

- ``API_KEY``: when set, prediction, forecast and explanation endpoints require
  a matching ``X-API-Key`` header. Health, version, metrics and docs stay open
  for load balancer checks and monitoring.
- ``RATE_LIMIT_PER_MINUTE``: when above zero, each client IP may make at most
  this many protected requests in any 60-second window.
- ``CORS_ALLOW_ORIGINS``: comma-separated browser origins allowed to call the
  API. Unset means no cross-origin browser access.
"""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

WINDOW_SECONDS = 60.0


class RateLimiter:
    """Sliding-window request counter per client, safe across worker threads."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client: str, limit: int, now: float | None = None) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def client_ip(request: Request) -> str:
    """Original client IP; behind the load balancer it is the first X-Forwarded-For entry."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def require_api_key(request: Request) -> None:
    """Reject the request unless it carries the configured API key."""
    expected = os.environ.get("API_KEY", "")
    if not expected:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def enforce_rate_limit(request: Request) -> None:
    """Reject the request with 429 once the client exceeds its per-minute budget."""
    limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "0") or 0)
    if not limiter.allow(client_ip(request), limit):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit of {limit} requests per minute exceeded.",
            headers={"Retry-After": str(int(WINDOW_SECONDS))},
        )


def configure_cors(app: FastAPI) -> None:
    """Allow the configured browser origins, if any."""
    origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )
