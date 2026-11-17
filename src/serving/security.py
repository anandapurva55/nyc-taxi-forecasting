"""Access controls for the prediction API: API key, rate limit and CORS.

Each control is configured by an environment variable so the same image runs
locally (all off) and in production (set by the ECS task definition):

- ``API_KEY``: when set, prediction, forecast, explanation, version and metrics
  endpoints require a matching ``X-API-Key`` header. ``/health`` stays open for
  load balancer checks, and ``/docs`` shows only the public request schema.
- ``RATE_LIMIT_PER_MINUTE``: when above zero, each client IP may make at most
  this many protected requests in any 60-second window.
- ``TRUSTED_PROXY_HOPS``: how many proxies in front of the API append to
  ``X-Forwarded-For`` (1 behind the load balancer, 2 behind CloudFront and the
  load balancer). The client IP is read that many entries from the right, so a
  caller cannot choose it by sending its own header.
- ``CORS_ALLOW_ORIGINS``: comma-separated browser origins allowed to call the
  API. Unset means no cross-origin browser access.
"""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

WINDOW_SECONDS = 60.0

# Declared as an OpenAPI security scheme so /docs offers an "Authorize" button.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Sliding-window request counter per client, safe across worker threads."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._next_sweep = 0.0

    def __len__(self) -> int:
        return len(self._hits)

    def _sweep(self, now: float) -> None:
        """Forget clients with no request in the last window, so memory stays bounded."""
        stale = [c for c, hits in self._hits.items() if not hits or now - hits[-1] >= WINDOW_SECONDS]
        for client in stale:
            del self._hits[client]
        self._next_sweep = now + WINDOW_SECONDS

    def allow(self, client: str, limit: int, now: float | None = None) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            if now >= self._next_sweep:
                self._sweep(now)
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


def trusted_proxy_hops() -> int:
    try:
        return max(0, int(os.environ.get("TRUSTED_PROXY_HOPS", "1")))
    except ValueError:
        return 1


def client_ip(request: Request) -> str:
    """Client IP as seen by the outermost trusted proxy.

    Each proxy appends the address it received the request from, so the entries
    a client sends itself sit on the left and cannot be trusted. With ``hops``
    trusted proxies, the client is the ``hops``-th entry from the right.
    """
    hops = trusted_proxy_hops()
    peer = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if not forwarded or hops == 0:
        return peer
    entries = [e.strip() for e in forwarded.split(",") if e.strip()]
    if len(entries) < hops:
        return entries[0] if entries else peer
    return entries[-hops]


def require_api_key(supplied: str | None = Security(api_key_header)) -> None:
    """Reject the request unless it carries the configured API key."""
    expected = os.environ.get("API_KEY", "")
    if not expected:
        return
    if not secrets.compare_digest((supplied or "").encode(), expected.encode()):
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
