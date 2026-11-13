import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mexc-scanner")

BASE = "https://contract.mexc.com"
REFRESH_SECONDS = int(os.getenv("REFRESH_SECONDS", "10"))
DEPTH_SYMBOL_LIMIT = int(os.getenv("DEPTH_SYMBOL_LIMIT", "30"))
MIN_SPREAD_DEFAULT = float(os.getenv("MIN_SPREAD_DEFAULT", "5"))
DEPTH_BANDS = (1, 3, 5, 10)

state: dict[str, Any] = {
    "updated_at": None,
    "contracts_total": 0,
    "rows": [],
    "status": "starting",
    "error": None,
    "refresh_seconds": REFRESH_SECONDS,
    "depth_symbol_limit": DEPTH_SYMBOL_LIMIT,
}
client: httpx.AsyncClient | None = None
scan_task: asyncio.Task | None = None
_rate_lock = asyncio.Lock()
_last_depth_request = 0.0


def as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


async def paced_depth_get(url: str) -> httpx.Response:
    global _last_depth_request
    async with _rate_lock:
        wait = 0.125 - (time.monotonic() - _last_depth_request)
        if wait > 0:
            await asyncio.sleep(wait)
        assert client is not None
        response = await client.get(url, timeout=12)
        _last_depth_request = time.monotonic()
        return response


def extract_data(payload: Any) -> Any:
    if isinstance(payload, dict):
        return payload.get("data", payload)
    return payload


async def fetch_depth(symbol: str, contract_size: float) -> dict[str, Any]:
    assert client is not None
    try:
        response = await paced_depth_get(f"{BASE}/api/v1/contract/depth/{symbol}")
        response.raise_for_status()
        payload = extract_data(response.json())
        bids = payload.get("bids", []) if isinstance(payload, dict) else []
        asks = payload.get("asks", []) if isinstance(payload, dict) else []
        return {"symbol": symbol, "bids": bids, "asks": asks, "contract_size": contract_size}
    except Exception as exc:
        log.warning("Depth failed for %s: %s", symbol, exc)
        return {"symbol": symbol, "bids": [], "asks": [], "contract_size": contract_size, "depth_error": str(exc)}


def depth_notional(levels: list, contract_size: float, side: str, mid: float, band_pct: float) -> float:
    total = 0.0
    for level in levels:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price, contracts = as_float(level[0]), as_float(level[1])
        if price <= 0 or contracts <= 0:
            continue
        if side == "asks" and price <= mid * (1 + band_pct / 100):
            total += price * contracts * contract_size
        elif side == "bids" and price >= mid * (1 - band_pct / 100):
            total += price * contracts * contract_size
    return total


async def scan_once() -> None:
    assert client is not None
    try:
        ticker_resp, detail_resp = await asyncio.gather(
            client.get(f"{BASE}/api/v1/contract/ticker", timeout=15),
            client.get(f"{BASE}/api/v1/contract/detail", timeout=15),
        )
        ticker_resp.raise_for_status()
        detail_resp.raise_for_status()
        tickers = extract_data(ticker_resp.json())
        details = extract_data(detail_resp.json())
        if isinstance(tickers, dict):
            tickers = [tickers]
        if not isinstance(tickers, list):
            raise RuntimeError("Unexpected ticker response format")

        meta: dict[str, dict[str, Any]] = {}
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict) and item.get("symbol"):
                    meta[item["symbol"]] = item

        rows = []
        for item in tickers:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol", ""))
            if not symbol.endswith("_USDT"):
                continue
            info = meta.get(symbol, {})
            # Keep only active perpetual contracts where contract metadata is available.
            if info and (info.get("state") not in (0, None) or info.get("futureType") not in (1, None)):
                continue
            bid = as_float(item.get("bid1"))
            ask = as_float(item.get("ask1"))
            if bid <= 0 or ask <= 0 or ask < bid:
                continue
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid * 100 if mid else 0
            rows.append({
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread,
                "last_price": as_float(item.get("lastPrice")),
                "volume24": as_float(item.get("volume24")),
                "amount24": as_float(item.get("amount24")),
                "open_interest_contracts": as_float(item.get("holdVol")),
                "fair_price": as_float(item.get("fairPrice")),
                "funding_rate": as_float(item.get("fundingRate")),
                "timestamp": item.get("timestamp"),
                "contract_size": as_float(info.get("contractSize")) or 1.0,
                "depth": None,
            })

        rows.sort(key=lambda r: r["spread_pct"], reverse=True)
        # Depth is the expensive part: calculate it for the widest-spread contracts.
        candidates = rows[:DEPTH_SYMBOL_LIMIT]
        semaphore = asyncio.Semaphore(3)

        async def get_one(row):
            async with semaphore:
                return await fetch_depth(row["symbol"], row["contract_size"])

        depth_results = await asyncio.gather(*(get_one(r) for r in candidates))
        depth_by_symbol = {d["symbol"]: d for d in depth_results}

        for row in rows:
            d = depth_by_symbol.get(row["symbol"])
            if not d or row["mid"] <= 0:
                continue
            row["depth"] = {}
            for band in DEPTH_BANDS:
                row["depth"][str(band)] = {
                    "bids_usdt": depth_notional(d["bids"], d["contract_size"], "bids", row["mid"], band),
                    "asks_usdt": depth_notional(d["asks"], d["contract_size"], "asks", row["mid"], band),
                }
            row["depth_error"] = d.get("depth_error")

        state.update({
            "updated_at": int(time.time() * 1000),
            "contracts_total": len(rows),
            "rows": rows,
            "status": "online",
            "error": None,
            "refresh_seconds": REFRESH_SECONDS,
            "depth_symbol_limit": DEPTH_SYMBOL_LIMIT,
        })
        log.info("Scan complete: %s USDT contracts; depth for %s", len(rows), len(candidates))
    except Exception as exc:
        log.exception("Scan failed")
        state["status"] = "error"
        state["error"] = str(exc)


async def scan_loop():
    while True:
        started = time.monotonic()
        await scan_once()
        elapsed = time.monotonic() - started
        await asyncio.sleep(max(1, REFRESH_SECONDS - elapsed))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, scan_task
    client = httpx.AsyncClient(headers={"User-Agent": "MEXC-Spread-Scanner/1.0"})
    scan_task = asyncio.create_task(scan_loop())
    yield
    if scan_task:
        scan_task.cancel()
        try:
            await scan_task
        except asyncio.CancelledError:
            pass
    await client.aclose()


app = FastAPI(title="MEXC Spread Scanner", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/scan")
async def get_scan(
    min_spread: float = Query(default=MIN_SPREAD_DEFAULT, ge=0, le=1000),
    sort: str = Query(default="spread"),
):
    rows = [r for r in state["rows"] if r["spread_pct"] >= min_spread]
    if sort == "volume":
        rows.sort(key=lambda r: r["amount24"], reverse=True)
    elif sort == "oi":
        rows.sort(key=lambda r: r["open_interest_contracts"], reverse=True)
    else:
        rows.sort(key=lambda r: r["spread_pct"], reverse=True)
    return {
        "updated_at": state["updated_at"],
        "contracts_total": state["contracts_total"],
        "count": len(rows),
        "status": state["status"],
        "error": state["error"],
        "refresh_seconds": state["refresh_seconds"],
        "depth_symbol_limit": state["depth_symbol_limit"],
        "rows": rows,
    }


@app.get("/health")
async def health():
    return {"ok": True, "status": state["status"]}
