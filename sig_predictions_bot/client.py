"""Minimal, dependency-free client for the Super Market API (/api/v1).

Handles the parts of the spec that are easy to get wrong:

* Auth: ``Authorization: Bearer <key>``, read from SUPER_MARKET_API_KEY.
* Rate limits: 100 reads and 30 writes per minute per account (shared by all
  of the account's keys). A client-side token bucket keeps us under both
  (with headroom), so we never hit 429 in normal operation.
* Retries: 429 waits for Retry-After; 503 TX_CONFLICT / SERVICE_UNAVAILABLE
  back off exponentially; 502 ORDER_STATUS_UNKNOWN and 409 REQUEST_IN_FLIGHT
  are retried with the same idempotency key, which the API guarantees never
  double-places. Other 4xx errors are not retried.
* Pagination: ``paginate`` follows ``pagination.nextCursor``.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_BASE = "https://www.thesuper.market/api/v1"


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.details = status, code, details


class TokenBucket:
    """Allows ``per_minute`` calls per rolling minute, refilled continuously."""

    def __init__(self, per_minute: float):
        self.capacity = per_minute
        self.tokens = per_minute
        self.rate = per_minute / 60.0
        self.last = time.monotonic()

    def take(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            time.sleep((1 - self.tokens) / self.rate)


class SuperMarketClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 reads_per_min: float = 90, writes_per_min: float = 27, timeout: float = 20):
        self.key = api_key or os.environ.get("SUPER_MARKET_API_KEY")
        if not self.key:
            raise RuntimeError("Set SUPER_MARKET_API_KEY (My Profile -> API Keys; scopes: read, trade)")
        self.base = (base_url or os.environ.get("SUPER_MARKET_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.reads = TokenBucket(reads_per_min)     # spec: 100/min; keep headroom
        self.writes = TokenBucket(writes_per_min)   # spec: 30/min
        self.timeout = timeout

    # ---- transport ------------------------------------------------------

    def request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None,
                body: Optional[Dict[str, Any]] = None, max_attempts: int = 5) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        data = json.dumps(body).encode() if body is not None else None
        bucket = self.reads if method in ("GET", "HEAD") else self.writes
        delay = 0.5
        for attempt in range(1, max_attempts + 1):
            bucket.take()
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                raw = e.read()
                try:
                    err = json.loads(raw).get("error", {})
                except (ValueError, AttributeError):
                    err = {}
                code = err.get("code", "HTTP_ERROR")
                if e.code == 207:  # batch with partial success is a valid result
                    return json.loads(raw)
                # 502 covers ORDER_STATUS_UNKNOWN, including batch bodies that
                # carry per-item results and no top-level code.
                retryable = (e.code in (429, 502) or code in ("TX_CONFLICT", "SERVICE_UNAVAILABLE",
                                                       "ORDER_STATUS_UNKNOWN", "REQUEST_IN_FLIGHT"))
                if not retryable or attempt == max_attempts:
                    raise ApiError(e.code, code, err.get("message", raw[:200]), err.get("details"))
                retry_after = e.headers.get("Retry-After")
                if code == "REQUEST_IN_FLIGHT":
                    wait = 90.0  # spec: the idempotency lease lasts 90 seconds
                elif retry_after:
                    wait = float(retry_after)
                else:
                    wait = delay + random.uniform(0, delay)
                    delay *= 2
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError):
                if attempt == max_attempts:
                    raise
                time.sleep(delay + random.uniform(0, delay))
                delay *= 2

    def get(self, path: str, **params) -> Any:
        return self.request("GET", path, params)

    def post(self, path: str, body: Dict[str, Any]) -> Any:
        return self.request("POST", path, body=body)

    def paginate(self, path: str, key: str = "data", max_pages: int = 50, **params) -> Iterator[dict]:
        cursor = None
        for _ in range(max_pages):
            page = self.get(path, cursor=cursor, **params)
            yield from page.get(key, [])
            pg = page.get("pagination") or {}
            cursor = pg.get("nextCursor")
            if not pg.get("hasMore") or not cursor:
                return

    # ---- endpoints used by the bot ----------------------------------------

    def tournament(self, slug: str) -> dict:
        return self.get(f"/tournaments/{slug}")

    def tournaments(self) -> List[dict]:
        r = self.get("/tournaments")
        return r.get("data", r) if isinstance(r, dict) else r

    def markets(self, tournament_id: Optional[str], status: str = "open") -> List[dict]:
        return list(self.paginate("/markets", limit=100, status=status, tournamentId=tournament_id))

    def market_orderbook(self, market_id: str, tournament_id: Optional[str], depth: int = 10) -> dict:
        return self.get(f"/markets/{market_id}/orderbook", tournamentId=tournament_id, depth=depth)

    def prices(self, exchange_ids: List[str], tournament_id: Optional[str]) -> List[dict]:
        out = []
        for i in range(0, len(exchange_ids), 100):  # endpoint takes at most 100 ids
            r = self.get("/exchanges/prices", ids=",".join(exchange_ids[i:i + 100]), tournamentId=tournament_id)
            out.extend(r.get("data", []))
        return out

    def relationships(self, tournament_id: Optional[str], rel_type: Optional[str] = None,
                      market_id: Optional[str] = None) -> List[dict]:
        return list(self.paginate("/relationships", limit=100, type=rel_type,
                                  tournamentId=tournament_id, marketId=market_id))

    def price_history(self, exchange_id: str, tournament_id: Optional[str], resolution: str = "1h",
                      start: Optional[str] = None, limit: int = 1000) -> dict:
        return self.get(f"/exchanges/{exchange_id}/price-history", tournamentId=tournament_id,
                        resolution=resolution, limit=limit, **({"from": start} if start else {}))

    def positions(self, slug: str) -> dict:
        return self.get(f"/tournaments/{slug}/portfolio/positions")

    def cancel_all(self, tournament_id: Optional[str]) -> Any:
        return self.post("/orders/cancel-all", {"tournamentId": tournament_id} if tournament_id else {})

    def place_batch(self, orders: List[dict]) -> Any:
        """Best-effort batch of up to 50 orders. One write against the rate limit."""
        return self.post("/orders/batch", {"idempotencyKey": str(uuid.uuid4()), "orders": orders})

    def place_multi_leg(self, legs: List[dict], relationship_id: Optional[str] = None) -> Any:
        """Atomic: every leg is placed or none is. Used for arbitrage baskets."""
        body: Dict[str, Any] = {"idempotencyKey": str(uuid.uuid4()), "legs": legs}
        if relationship_id:
            body["relationshipConstraint"] = relationship_id
        return self.post("/orders/multi-leg", body)
