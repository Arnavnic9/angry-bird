"""Runs the live bot against an in-process fake of the Super Market API.

The fake returns responses shaped like the OpenAPI spec and records every
write. That checks the whole live loop end to end (universe -> portfolio ->
screen -> books -> decide -> cancel-all -> multi-leg / batch) without a key.
"""

import json
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

from sig_predictions_bot.bot import Bot
from sig_predictions_bot.client import SuperMarketClient
from sig_predictions_bot.config import StrategyParams

TID = "550e8400-e29b-41d4-a716-446655440000"

MARKETS = [
    # Binary market with a wide spread: the bot should quote it.
    {"id": "1", "title": "Binary", "status": "open", "settlementDate": "2099-01-01T00:00:00Z",
     "isMultiOutcome": False, "exchanges": [{"id": "10", "option": "YES"}]},
    # Three-way exclusive, exhaustive market whose asks sum to 0.90: free money.
    {"id": "2", "title": "Three-way", "status": "open", "settlementDate": "2099-01-01T00:00:00Z",
     "isMultiOutcome": True, "exchanges": [{"id": "20", "option": "A"}, {"id": "21", "option": "B"},
                                           {"id": "22", "option": "C"}]},
]
BOOKS = {
    "1": {"exchanges": [{"exchangeId": "10", "bids": [{"price": 0.40, "quantity": 200}],
                         "asks": [{"price": 0.50, "quantity": 200}]}]},
    "2": {"exchanges": [{"exchangeId": e, "bids": [{"price": 0.25, "quantity": 100}],
                         "asks": [{"price": 0.30, "quantity": 100}]} for e in ("20", "21", "22")],
          "overround": 0.825, "hasArbitrageOpportunity": True},
}


class FakeApi(BaseHTTPRequestHandler):
    writes = []

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        assert self.headers["Authorization"] == "Bearer test-key"
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        path = u.path.replace("/api/v1", "")
        if path == "/tournaments/cup":
            return self._send({"id": TID, "slug": "cup", "myBalance": 10000.0})
        if path == "/markets":
            return self._send({"data": MARKETS, "pagination": {"hasMore": False, "nextCursor": None}})
        if path == "/relationships":
            return self._send({"data": [{"id": "rel-1", "type": "mutually_exclusive", "status": "active",
                                         "isExhaustive": True,
                                         "nodes": [{"exchangeId": e} for e in ("20", "21", "22")]}],
                               "pagination": {"hasMore": False, "nextCursor": None}})
        if path == "/tournaments/cup/portfolio/positions":
            return self._send({"positions": [], "summary": {}})
        if path == "/exchanges/prices":
            ids = q["ids"][0].split(",")
            data = []
            for m in BOOKS.values():
                for e in m["exchanges"]:
                    if e["exchangeId"] in ids:
                        bb, ba = e["bids"][0]["price"], e["asks"][0]["price"]
                        data.append({"exchangeId": e["exchangeId"], "latestPrice": (bb + ba) / 2,
                                     "bestBid": bb, "bestAsk": ba, "spread": round(ba - bb, 3)})
            return self._send({"data": data, "missingIds": []})
        if path.startswith("/markets/") and path.endswith("/orderbook"):
            return self._send(BOOKS[path.split("/")[2]])
        self._send({"error": {"code": "NOT_FOUND", "message": path}}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        path = urllib.parse.urlparse(self.path).path.replace("/api/v1", "")
        FakeApi.writes.append((path, body))
        if path == "/orders/batch":
            return self._send({"results": [{"index": i, "ok": True, "status": 200, "data": {}}
                                           for i in range(len(body["orders"]))]})
        self._send({"ok": True})


class BotLoopTest(unittest.TestCase):
    def test_one_live_cycle_against_fake_api(self):
        srv = HTTPServer(("127.0.0.1", 0), FakeApi)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            client = SuperMarketClient("test-key", f"http://127.0.0.1:{srv.server_port}/api/v1",
                                       reads_per_min=10_000, writes_per_min=10_000)
            bot = Bot(client, "cup", StrategyParams(), live=True, max_books=10, max_drawdown=0.5)
            bot.run(cycle_seconds=0, max_cycles=1)
        finally:
            srv.shutdown()
            srv.server_close()

        paths = [p for p, _ in FakeApi.writes]
        self.assertEqual(paths[0], "/orders/cancel-all")            # stale quotes cleared first
        legs = [b for p, b in FakeApi.writes if p == "/orders/multi-leg"]
        self.assertEqual(len(legs), 1)                               # the arb basket, atomically
        self.assertEqual({l["exchangeId"] for l in legs[0]["legs"]}, {"20", "21", "22"})
        self.assertTrue(all(l["side"] == "yes" and l["action"] == "buy" and l["tournamentId"] == TID
                            for l in legs[0]["legs"]))
        self.assertNotIn("relationshipConstraint", legs[0])
        batches = [b for p, b in FakeApi.writes if p == "/orders/batch"]
        self.assertTrue(batches and batches[0]["idempotencyKey"])
        quoted = {o["exchangeId"] for o in batches[0]["orders"]}
        self.assertIn("10", quoted)                                  # wide binary book gets quoted
        for o in batches[0]["orders"]:
            self.assertAlmostEqual(o["price"] / 0.005, round(o["price"] / 0.005), places=6)
            self.assertTrue(0.005 <= o["price"] <= 0.995)


if __name__ == "__main__":
    unittest.main()
