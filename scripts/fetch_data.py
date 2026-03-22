#!/usr/bin/env python3
"""
Automated data fetcher for the Trading Bot.

Fetches OHLCV and alternative data from multiple sources and saves to
data/raw/ as both CSV and Parquet.  Supports incremental updates
(only fetches missing bars).

Usage:
    python scripts/fetch_data.py --asset BTCUSDT --period 2y --interval 1h
    python scripts/fetch_data.py --asset BTCUSDT --period 1y --interval 1d --cross-assets
    python scripts/fetch_data.py --asset BTCUSDT --interval 1h --schedule daily

Sources:
    - CCXT (Binance/Bybit):  OHLCV + funding rate
    - yfinance:              SPY, DXY (UUP), VIX, Gold, US 10Y
    - alternative.me:        Fear & Greed Index (crypto)
    - CoinGecko (free tier): on-chain proxy metrics

Flags:
    --asset     Symbol, e.g. BTCUSDT or BTC/USDT
    --period    History length: 1d, 1w, 1m, 3m, 6m, 1y, 2y, 3y
    --interval  Bar size: 1m, 5m, 15m, 1h, 4h, 1d
    --exchange  CCXT exchange id (default: binance)
    --cross-assets  Also fetch SPY, DXY, VIX, Gold cross-asset data
    --alt-data      Also fetch Fear&Greed + CoinGecko on-chain proxy
    --schedule  [daily] Set up scheduling instructions
    --output    Output directory (default: data/raw)
    --no-parquet  Skip Parquet output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ── CCXT ──────────────────────────────────────────────────────────────
try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False

# ── yfinance ──────────────────────────────────────────────────────────
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# ── requests (alternative data) ───────────────────────────────────────
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("fetch_data")

# ─────────────────────────────────────────────
# Period → timedelta mapping
# ─────────────────────────────────────────────

PERIOD_MAP: Dict[str, timedelta] = {
    "1d":  timedelta(days=1),
    "1w":  timedelta(weeks=1),
    "1m":  timedelta(days=30),
    "3m":  timedelta(days=90),
    "6m":  timedelta(days=180),
    "1y":  timedelta(days=365),
    "2y":  timedelta(days=730),
    "3y":  timedelta(days=1095),
}

# yfinance interval aliases
YF_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "1h",   # yf has no 4h; use 1h and resample downstream
    "1d": "1d",
}

# CCXT timeframe aliases
CCXT_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}

# ─────────────────────────────────────────────
# Week 27/28: Public Python API helpers
# run_full_pipeline.py 및 테스트에서 직접 import해 사용
# ─────────────────────────────────────────────

def _period_to_start_date(period: str) -> datetime:
    """
    'Xy' / 'Xm' / 'Xd' 형식의 문자열을 datetime (UTC-naive)으로 변환합니다.

    PERIOD_MAP 단축 키도 지원:
        '1y', '2y', '3m', '6m' 등

    Examples:
        _period_to_start_date('2y')  → 2년 전
        _period_to_start_date('6m')  → 6개월 전
        _period_to_start_date('30d') → 30일 전
    """
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    if period in PERIOD_MAP:
        return now - PERIOD_MAP[period]
    unit = period[-1].lower()
    try:
        value = int(period[:-1])
    except ValueError:
        raise ValueError(f"Unknown period format: {period!r}. Use e.g. '2y', '6m', '30d'")
    if unit == "y":
        return now - timedelta(days=365 * value)
    if unit == "m":
        return now - timedelta(days=30 * value)
    if unit == "d":
        return now - timedelta(days=value)
    raise ValueError(f"Unknown period unit {unit!r} in {period!r}. Use y/m/d.")


def fetch_data(
    asset: str = "BTCUSDT",
    period: str = "2y",
    interval: str = "1h",
    output: Optional[str] = None,
    source: str = "auto",
    exchange: str = "binance",
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Python API wrapper — fetch_data.py 스크립트의 핵심 기능을 함수로 노출합니다.
    run_full_pipeline.py 및 테스트에서 직접 import해 사용합니다.

    Args:
        asset:    심볼 (e.g. 'BTCUSDT', 'BTC/USDT', 'SPY')
        period:   기간 (e.g. '2y', '6m', '30d')
        interval: 타임프레임 (e.g. '1h', '1d')
        output:   저장 경로 (.csv). None이면 저장 안 함.
        source:   'ccxt' | 'yfinance' | 'auto'
        exchange: CCXT 거래소 ID
        dry_run:  True → 실제 다운로드 없이 synthetic 데이터 반환

    Returns:
        pd.DataFrame with columns [$open, $high, $low, $close, $volume]
    """
    if dry_run:
        start_dt = _period_to_start_date(period)
        dates = pd.date_range(start_dt, datetime.utcnow(), freq="h", tz="UTC")
        n = min(len(dates), 200)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "$open":   rng.uniform(30000, 50000, n),
            "$high":   rng.uniform(30000, 50000, n),
            "$low":    rng.uniform(30000, 50000, n),
            "$close":  rng.uniform(30000, 50000, n),
            "$volume": rng.uniform(100, 10000, n),
        }, index=dates[:n])
        logger.info("[DRY RUN] Returning %d synthetic rows for %s", n, asset)
        if output:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output)
        return df

    # Map period to PERIOD_MAP key
    if period in PERIOD_MAP:
        _period_key = period
    else:
        start = _period_to_start_date(period)
        days = (datetime.utcnow() - start).days
        _period_key = min(PERIOD_MAP.keys(), key=lambda k: abs(PERIOD_MAP[k].days - days))

    now = datetime.now(tz=timezone.utc)
    since = now - PERIOD_MAP[_period_key]
    since_ms = int(since.timestamp() * 1000)

    # Auto-detect source
    if source == "auto":
        source = "ccxt" if any(asset.upper().endswith(s) for s in ("USDT", "BTC", "ETH", "BNB")) else "yfinance"

    out_dir = Path(output).parent if output else Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "ccxt" and _CCXT_AVAILABLE:
        df = fetch_ccxt(
            symbol=asset,
            exchange_id=exchange,
            timeframe=CCXT_TIMEFRAME_MAP.get(interval, "1h"),
            since_ms=since_ms,
            output_dir=out_dir,
            no_parquet=True,
        )
    elif _YF_AVAILABLE:
        df = fetch_yfinance(
            tickers=[asset],
            since=since,
            interval=YF_INTERVAL_MAP.get(interval, "1h"),
            output_dir=out_dir,
            no_parquet=True,
        )
        if df is not None and not df.empty:
            df = df.rename(columns={c: f"${c.lower()}" for c in df.columns if not c.startswith("$")})
    else:
        raise RuntimeError("Neither ccxt nor yfinance is installed.")

    if df is None or df.empty:
        raise RuntimeError(f"No data fetched for {asset}")

    if output:
        df.to_csv(output)

    return df


# ─────────────────────────────────────────────
# Fetch Log (incremental update tracking)
# ─────────────────────────────────────────────

def _load_fetch_log(log_path: Path) -> dict:
    if log_path.exists():
        try:
            return json.loads(log_path.read_text())
        except Exception:
            pass
    return {}


def _save_fetch_log(log_path: Path, log: dict) -> None:
    log_path.write_text(json.dumps(log, indent=2, default=str))


# ─────────────────────────────────────────────
# Data integrity helpers
# ─────────────────────────────────────────────

def _validate_ohlcv(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Check for NaN gaps and basic OHLCV sanity."""
    nan_rows = df.isnull().any(axis=1).sum()
    if nan_rows > 0:
        logger.warning(f"[{name}] {nan_rows} 행에 NaN 존재 — forward fill 적용")
        df = df.ffill().bfill()

    # high >= low sanity
    if "$high" in df.columns and "$low" in df.columns:
        bad = (df["$high"] < df["$low"]).sum()
        if bad > 0:
            logger.warning(f"[{name}] {bad} 행에서 high < low — swap 적용")
            mask = df["$high"] < df["$low"]
            df.loc[mask, ["$high", "$low"]] = df.loc[mask, ["$low", "$high"]].values

    return df


def _save(df: pd.DataFrame, path_stem: Path, no_parquet: bool = False) -> None:
    """Save DataFrame as CSV (and optionally Parquet)."""
    csv_path = path_stem.with_suffix(".csv")
    df.to_csv(csv_path)
    logger.info(f"  저장: {csv_path}  ({len(df)} rows)")

    if not no_parquet:
        try:
            pq_path = path_stem.with_suffix(".parquet")
            df.to_parquet(pq_path, engine="pyarrow")
            logger.info(f"  저장: {pq_path}")
        except ImportError:
            logger.debug("pyarrow 없음 — Parquet 건너뜀")


# ─────────────────────────────────────────────
# 1. CCXT — crypto OHLCV + funding rate
# ─────────────────────────────────────────────

def fetch_ccxt(
    symbol: str,
    exchange_id: str,
    timeframe: str,
    since_ms: int,
    output_dir: Path,
    no_parquet: bool,
) -> Optional[pd.DataFrame]:
    if not _CCXT_AVAILABLE:
        logger.error("ccxt 미설치 — pip install ccxt")
        return None

    # Normalise symbol format
    sym_ccxt = symbol.replace("USDT", "/USDT").replace("//", "/")
    if "/" not in sym_ccxt:
        sym_ccxt = f"{sym_ccxt[:3]}/{sym_ccxt[3:]}"

    logger.info(f"[CCXT] {exchange_id} / {sym_ccxt} / {timeframe}")
    try:
        ex_class = getattr(ccxt, exchange_id)
        exchange = ex_class({"enableRateLimit": True})
    except AttributeError:
        logger.error(f"지원하지 않는 거래소: {exchange_id}")
        return None

    all_rows = []
    since = since_ms
    limit = 1000

    while True:
        try:
            bars = exchange.fetch_ohlcv(sym_ccxt, timeframe, since=since, limit=limit)
        except ccxt.NetworkError as e:
            logger.warning(f"네트워크 오류: {e} — 5초 후 재시도")
            time.sleep(5)
            continue
        except ccxt.BaseError as e:
            # Try fallback exchange
            if exchange_id == "binance":
                logger.warning(f"Binance 오류: {e} — Bybit으로 fallback")
                try:
                    exchange = ccxt.bybit({"enableRateLimit": True})
                    bars = exchange.fetch_ohlcv(sym_ccxt, timeframe, since=since, limit=limit)
                except Exception as e2:
                    logger.error(f"Bybit도 실패: {e2}")
                    return None
            else:
                logger.error(f"CCXT 오류: {e}")
                return None

        if not bars:
            break
        all_rows.extend(bars)
        last_ts = bars[-1][0]
        if len(bars) < limit:
            break
        since = last_ts + 1
        time.sleep(exchange.rateLimit / 1000)

    if not all_rows:
        logger.error("데이터 없음")
        return None

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "$open", "$high", "$low", "$close", "$volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df = _validate_ohlcv(df, sym_ccxt)

    safe_name = symbol.replace("/", "").upper()
    _save(df, output_dir / f"{safe_name}_{timeframe}", no_parquet)

    # Funding rate (if available, for perpetual futures)
    _fetch_funding_rate(exchange, sym_ccxt, since_ms, output_dir, safe_name, no_parquet)

    return df


def _fetch_funding_rate(
    exchange,
    symbol: str,
    since_ms: int,
    output_dir: Path,
    safe_name: str,
    no_parquet: bool,
) -> None:
    """Fetch funding rate history if the exchange/symbol supports it."""
    if not hasattr(exchange, "fetch_funding_rate_history"):
        return
    try:
        rows = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=500)
        if not rows:
            return
        fr = pd.DataFrame(rows)
        fr["timestamp"] = pd.to_datetime(fr["timestamp"], unit="ms", utc=True)
        fr = fr.set_index("timestamp")[["fundingRate"]].rename(
            columns={"fundingRate": "funding_rate"}
        )
        _save(fr, output_dir / f"{safe_name}_funding_rate", no_parquet)
        logger.info(f"  Funding rate: {len(fr)} rows")
    except Exception as e:
        logger.debug(f"Funding rate fetch 실패 (무시): {e}")


# ─────────────────────────────────────────────
# 2. yfinance — cross-asset data
# ─────────────────────────────────────────────

_YF_TICKERS = {
    "SPY":    "SPY",       # S&P 500 ETF
    "DXY":    "DX-Y.NYB",  # US Dollar Index (Yahoo)
    "VIX":    "^VIX",      # CBOE Volatility Index
    "GOLD":   "GC=F",      # Gold Futures
    "US10Y":  "^TNX",      # US 10Y Treasury Yield
    "ETH":    "ETH-USD",   # Ethereum (for crypto cross-asset)
}


def fetch_yfinance(
    tickers: list[str],
    since: datetime,
    interval: str,
    output_dir: Path,
    no_parquet: bool,
) -> Dict[str, pd.DataFrame]:
    if not _YF_AVAILABLE:
        logger.warning("yfinance 미설치 — pip install yfinance  (cross-asset 건너뜀)")
        return {}

    yf_interval = YF_INTERVAL_MAP.get(interval, "1d")
    results = {}

    for name in tickers:
        ticker_sym = _YF_TICKERS.get(name.upper(), name)
        logger.info(f"[yfinance] {name} ({ticker_sym}) / {yf_interval}")
        try:
            tkr = yf.Ticker(ticker_sym)
            df = tkr.history(
                start=since.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning(f"  {name}: 데이터 없음")
                continue

            df.index = pd.to_datetime(df.index, utc=True)
            # Rename to standard $open/$high/$low/$close/$volume
            col_map = {
                "Open": "$open", "High": "$high", "Low": "$low",
                "Close": "$close", "Volume": "$volume",
            }
            df = df.rename(columns=col_map)
            keep = [c for c in ["$open", "$high", "$low", "$close", "$volume"] if c in df.columns]
            df = df[keep]
            df = _validate_ohlcv(df, name)

            _save(df, output_dir / f"{name.upper()}_{yf_interval}", no_parquet)
            results[name] = df
        except Exception as e:
            logger.warning(f"  {name} 수집 실패: {e}")

    return results


# ─────────────────────────────────────────────
# 3. Fear & Greed Index (alternative.me)
# ─────────────────────────────────────────────

def fetch_fear_greed(output_dir: Path, no_parquet: bool, limit: int = 365) -> Optional[pd.DataFrame]:
    if not _REQUESTS_AVAILABLE:
        logger.warning("requests 미설치 — Fear & Greed 건너뜀")
        return None

    logger.info("[Fear & Greed] alternative.me API")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if not data:
            logger.warning("Fear & Greed: 빈 응답")
            return None

        df = pd.DataFrame(data)[["timestamp", "value", "value_classification"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["value"] = pd.to_numeric(df["value"])
        # Normalize to [-1, 1]
        df["fear_greed_norm"] = (df["value"] / 50.0) - 1.0
        df = df.set_index("timestamp").sort_index()

        _save(df, output_dir / "fear_greed", no_parquet)
        logger.info(f"  Fear & Greed: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  Fear & Greed 수집 실패: {e}")
        return None


# ─────────────────────────────────────────────
# 4. CoinGecko on-chain proxy metrics (free tier)
# ─────────────────────────────────────────────

def fetch_coingecko_market(
    coin_id: str,
    days: int,
    output_dir: Path,
    no_parquet: bool,
) -> Optional[pd.DataFrame]:
    if not _REQUESTS_AVAILABLE:
        return None

    logger.info(f"[CoinGecko] {coin_id} market chart ({days} days)")
    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        f"?vs_currency=usd&days={days}&interval=daily"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            logger.warning("CoinGecko rate limit — 60초 대기 후 재시도")
            time.sleep(60)
            resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        prices  = pd.DataFrame(data["prices"],        columns=["ts", "price"])
        mcaps   = pd.DataFrame(data["market_caps"],   columns=["ts", "market_cap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "volume"])

        df = prices.merge(mcaps, on="ts").merge(volumes, on="ts")
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.drop(columns=["ts"]).set_index("timestamp").sort_index()

        # Simple derived on-chain proxy: NVT ratio approximation
        df["nvt_proxy"] = np.where(
            df["volume"] > 0,
            np.tanh(np.log1p(df["market_cap"]) - np.log1p(df["volume"])),
            0.0,
        )

        _save(df, output_dir / f"{coin_id}_coingecko", no_parquet)
        logger.info(f"  CoinGecko {coin_id}: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"  CoinGecko 수집 실패 (graceful degradation): {e}")
        return None


# ─────────────────────────────────────────────
# Scheduling helper
# ─────────────────────────────────────────────

def print_schedule_instructions(script_path: Path, args: argparse.Namespace) -> None:
    cmd = (
        f"python {script_path} "
        f"--asset {args.asset} "
        f"--interval {args.interval} "
        f"{'--cross-assets ' if args.cross_assets else ''}"
        f"{'--alt-data ' if args.alt_data else ''}"
    ).strip()

    print("\n" + "="*60)
    print("  자동 갱신 설정 (매일 자정)")
    print("="*60)

    # macOS / Linux cron
    print("\n▶ macOS / Linux (crontab -e):")
    print(f"  0 0 * * * cd {script_path.parent.parent} && {cmd} >> logs/fetch_data.log 2>&1")

    # Windows Task Scheduler
    print("\n▶ Windows (Task Scheduler):")
    print(f'  Program: python')
    print(f'  Arguments: {script_path} --asset {args.asset} --interval {args.interval}')
    print(f'  Start in: {script_path.parent.parent}')
    print(f'  Trigger: Daily at 00:00')
    print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trading Bot — automated data fetcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--asset",       default="BTCUSDT",  help="심볼 (e.g. BTCUSDT, ETH/USDT)")
    parser.add_argument("--period",      default="1y",       choices=list(PERIOD_MAP), help="수집 기간")
    parser.add_argument("--interval",    default="1h",       choices=list(CCXT_TIMEFRAME_MAP), help="봉 단위")
    parser.add_argument("--exchange",    default="binance",  help="CCXT 거래소 ID")
    parser.add_argument("--cross-assets", action="store_true", help="SPY, DXY, VIX, Gold, US10Y 수집")
    parser.add_argument("--alt-data",    action="store_true", help="Fear&Greed + CoinGecko 수집")
    parser.add_argument("--schedule",    choices=["daily"],  help="스케줄 설정 가이드 출력")
    parser.add_argument("--output",      default="data/raw", help="저장 디렉토리")
    parser.add_argument("--no-parquet",  action="store_true", help="Parquet 저장 건너뜀")
    parser.add_argument("--dry-run",     action="store_true", help="실제 다운로드 없이 synthetic 데이터 반환 (테스트용)")
    args = parser.parse_args()

    # ── Dry-run shortcut ──
    if args.dry_run:
        df = fetch_data(asset=args.asset, period=args.period, interval=args.interval, dry_run=True)
        logger.info("[DRY RUN] %d rows — columns: %s", len(df), list(df.columns))
        return

    # ── Paths ──
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / ".fetch_log.json"
    fetch_log = _load_fetch_log(log_path)

    # ── Time range ──
    now = datetime.now(tz=timezone.utc)
    period_delta = PERIOD_MAP[args.period]
    default_since = now - period_delta

    # Incremental: use last fetch time if available
    asset_key = f"{args.asset}_{args.interval}"
    last_fetch_str = fetch_log.get(asset_key)
    if last_fetch_str:
        last_fetch = datetime.fromisoformat(last_fetch_str)
        if last_fetch > default_since:
            logger.info(f"[증분 업데이트] 마지막 수집: {last_fetch.strftime('%Y-%m-%d %H:%M')} UTC")
            since = last_fetch - timedelta(hours=1)   # 약간 겹치게 해서 gap 방지
        else:
            since = default_since
    else:
        since = default_since

    since_ms = int(since.timestamp() * 1000)
    logger.info(f"수집 시작: {since.strftime('%Y-%m-%d %H:%M')} UTC → {now.strftime('%Y-%m-%d %H:%M')} UTC")

    # ── 1. Primary asset OHLCV ──
    logger.info(f"\n{'─'*50}")
    logger.info(f"Primary: {args.asset} / {args.interval} / {args.exchange}")
    df_main = fetch_ccxt(
        symbol=args.asset,
        exchange_id=args.exchange,
        timeframe=CCXT_TIMEFRAME_MAP[args.interval],
        since_ms=since_ms,
        output_dir=output_dir,
        no_parquet=args.no_parquet,
    )

    if df_main is not None:
        fetch_log[asset_key] = now.isoformat()
        logger.info(f"  Primary 수집 완료: {len(df_main)} rows")

    # ── 2. Cross-asset ──
    if args.cross_assets:
        logger.info(f"\n{'─'*50}")
        logger.info("Cross-asset: SPY, DXY, VIX, GOLD, US10Y, ETH")
        fetch_yfinance(
            tickers=["SPY", "DXY", "VIX", "GOLD", "US10Y", "ETH"],
            since=since,
            interval=args.interval,
            output_dir=output_dir,
            no_parquet=args.no_parquet,
        )
        fetch_log["cross_assets"] = now.isoformat()

    # ── 3. Alternative data ──
    if args.alt_data:
        logger.info(f"\n{'─'*50}")
        logger.info("Alternative data: Fear & Greed + CoinGecko")

        days = min(int(period_delta.days), 365)
        fetch_fear_greed(output_dir, args.no_parquet, limit=days)

        # CoinGecko — detect coin from asset
        coin_map = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "SOLUSDT": "solana"}
        coin_id = coin_map.get(args.asset.upper(), "bitcoin")
        fetch_coingecko_market(coin_id, days, output_dir, args.no_parquet)
        fetch_log["alt_data"] = now.isoformat()

    # ── Save fetch log ──
    _save_fetch_log(log_path, fetch_log)

    # ── Scheduling instructions ──
    if args.schedule == "daily":
        print_schedule_instructions(Path(__file__).resolve(), args)

    # ── Summary ──
    logger.info(f"\n{'='*50}")
    logger.info("수집 완료")
    logger.info(f"저장 위치: {output_dir}")
    files = sorted(output_dir.glob("*.csv"))
    for f in files:
        size_kb = f.stat().st_size // 1024
        logger.info(f"  {f.name:<40} {size_kb:>6} KB")


if __name__ == "__main__":
    main()
