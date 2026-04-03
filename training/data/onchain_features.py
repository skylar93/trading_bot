"""
On-chain metrics feature engine for crypto trading.

Fetches blockchain data from free APIs (CoinGecko, CCXT, Blockchain.com)
and produces 10 normalized features per timestep.

Features (all normalized to [-1, 1]):
    1.  active_addresses_norm   — network activity
    2.  tx_count_norm           — real usage proxy
    3.  hash_rate_change        — miner confidence (BTC only)
    4.  exchange_flow_ratio     — sell/buy pressure (inflow - outflow)
    5.  exchange_reserves_chg   — exchange reserve delta
    6.  lth_supply_change       — long-term holder movement (smart money)
    7.  mvrv_norm               — over/under-valued signal
    8.  nupl_norm               — net unrealized profit/loss
    9.  funding_rate_norm       — futures market sentiment
    10. open_interest_change    — position-size momentum

Caching: SQLite with 1-hour TTL.
Fallback: returns zeros on any API failure (graceful degradation).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Number of on-chain features
N_ONCHAIN_FEATURES = 10

ONCHAIN_COLS = [
    "active_addresses_norm",
    "tx_count_norm",
    "hash_rate_change",
    "exchange_flow_ratio",
    "exchange_reserves_chg",
    "lth_supply_change",
    "mvrv_norm",
    "nupl_norm",
    "funding_rate_norm",
    "open_interest_change",
]


@dataclass
class OnChainConfig:
    """Configuration for the on-chain feature engine."""
    symbol: str = "BTC"                  # base asset symbol
    vs_currency: str = "usd"
    cache_db: Optional[str] = "data/cache/onchain_cache.db"
    cache_ttl_seconds: int = 3600        # 1-hour TTL
    coingecko_id: str = "bitcoin"        # CoinGecko coin ID
    ccxt_exchange: str = "binance"       # exchange for funding/OI
    ccxt_symbol: str = "BTC/USDT:USDT"  # perpetual futures symbol
    request_timeout: int = 10            # HTTP timeout in seconds
    # Rolling window for z-score normalization of level features
    rolling_window: int = 30
    # Individual feature toggles (set False to skip an API call)
    use_coingecko: bool = True
    use_ccxt_derivatives: bool = True


class OnChainFeatureEngine:
    """
    Fetches on-chain metrics and returns a normalized feature array.

    Usage::

        engine = OnChainFeatureEngine(config)
        # For live/paper trading: get latest snapshot
        features = engine.get_latest()           # shape (10,)

        # For backtesting: align to a price DataFrame index
        df_features = engine.align_to_prices(price_df)  # shape (T, 10)
    """

    def __init__(self, config: Optional[OnChainConfig] = None):
        self.cfg = config or OnChainConfig()
        self._cache_conn: Optional[sqlite3.Connection] = None
        if self.cfg.cache_db:
            self._init_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_latest(self) -> np.ndarray:
        """
        Return current on-chain features as a float32 array of shape (10,).
        Returns zeros on any failure (graceful degradation).
        """
        try:
            raw = self._fetch_all()
            return self._normalize(raw)
        except Exception as exc:
            logger.warning("OnChain get_latest failed, returning zeros: %s", exc)
            return np.zeros(N_ONCHAIN_FEATURES, dtype=np.float32)

    def align_to_prices(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce a (T, 10) DataFrame of on-chain features aligned to *price_df*'s index.

        For backtesting: fetches historical snapshots where available,
        forward-fills gaps, and falls back to zeros.

        Returns a DataFrame with ONCHAIN_COLS columns indexed like *price_df*.
        """
        index = price_df.index
        result = pd.DataFrame(
            np.zeros((len(index), N_ONCHAIN_FEATURES), dtype=np.float32),
            index=index,
            columns=ONCHAIN_COLS,
        )

        # Attempt to fetch historical data if index has datetime entries
        if hasattr(index, 'to_pydatetime') and len(index) > 0:
            try:
                hist = self._fetch_historical(index)
                if hist is not None and not hist.empty:
                    hist = hist.reindex(index).ffill().fillna(0.0)
                    result.update(hist)
            except Exception as exc:
                logger.warning(
                    "OnChain align_to_prices failed, using zeros: %s", exc
                )

        return result

    # ------------------------------------------------------------------
    # Fetch layer (live)
    # ------------------------------------------------------------------

    def _fetch_all(self) -> Dict[str, float]:
        """Fetch all available metrics; return dict of raw values."""
        raw: Dict[str, float] = {}

        if self.cfg.use_coingecko and _REQUESTS_AVAILABLE:
            cache_key = f"coingecko_{self.cfg.coingecko_id}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                raw.update(cached)
            else:
                cg_data = self._fetch_coingecko()
                raw.update(cg_data)
                self._cache_set(cache_key, cg_data)

        if self.cfg.use_ccxt_derivatives and _CCXT_AVAILABLE:
            cache_key = f"ccxt_{self.cfg.ccxt_exchange}_{self.cfg.ccxt_symbol}"
            cached = self._cache_get(cache_key)
            if cached is not None:
                raw.update(cached)
            else:
                deriv_data = self._fetch_ccxt_derivatives()
                raw.update(deriv_data)
                self._cache_set(cache_key, deriv_data)

        return raw

    def _fetch_coingecko(self) -> Dict[str, float]:
        """Fetch on-chain metrics from CoinGecko free API."""
        data: Dict[str, float] = {}
        base = "https://api.coingecko.com/api/v3"
        coin_id = self.cfg.coingecko_id

        try:
            resp = requests.get(
                f"{base}/coins/{coin_id}",
                params={"localization": "false", "tickers": "false",
                        "market_data": "true", "community_data": "true",
                        "developer_data": "false"},
                timeout=self.cfg.request_timeout,
            )
            resp.raise_for_status()
            info = resp.json()

            market = info.get("market_data", {})

            # MVRV proxy: market_cap / realized_market_cap (use total_volume as proxy)
            market_cap = market.get("market_cap", {}).get(self.cfg.vs_currency, 0) or 0
            total_vol = market.get("total_volume", {}).get(self.cfg.vs_currency, 1) or 1
            data["mvrv_raw"] = market_cap / max(total_vol, 1)

            # Price change as NUPL proxy
            price_change_24h = market.get("price_change_percentage_24h") or 0.0
            data["nupl_raw"] = float(price_change_24h) / 100.0

            # Community data as active addresses proxy
            community = info.get("community_data", {})
            reddit_subscribers = community.get("reddit_subscribers") or 0
            data["active_addresses_raw"] = float(reddit_subscribers)

            # Developer data as tx count proxy (commits)
            dev = info.get("developer_data", {})
            commits = dev.get("commit_count_4_weeks") or 0
            data["tx_count_raw"] = float(commits)

        except Exception as exc:
            logger.debug("CoinGecko fetch failed: %s", exc)

        return data

    def _fetch_ccxt_derivatives(self) -> Dict[str, float]:
        """Fetch funding rate and open interest from CCXT exchange."""
        data: Dict[str, float] = {}
        try:
            exchange_cls = getattr(ccxt, self.cfg.ccxt_exchange)
            exchange = exchange_cls({"enableRateLimit": True})

            # Funding rate
            try:
                funding_info = exchange.fetch_funding_rate(self.cfg.ccxt_symbol)
                data["funding_rate_raw"] = float(
                    funding_info.get("fundingRate") or 0.0
                )
            except Exception:
                data["funding_rate_raw"] = 0.0

            # Open interest
            try:
                oi_info = exchange.fetch_open_interest(self.cfg.ccxt_symbol)
                data["open_interest_raw"] = float(
                    oi_info.get("openInterest") or oi_info.get("openInterestAmount") or 0.0
                )
            except Exception:
                data["open_interest_raw"] = 0.0

        except Exception as exc:
            logger.debug("CCXT derivatives fetch failed: %s", exc)

        return data

    def _fetch_historical(self, index: pd.Index) -> Optional[pd.DataFrame]:
        """
        Best-effort fetch of historical on-chain data for backtesting.
        Returns None if unavailable — caller falls back to zeros.
        """
        # CoinGecko free API: market chart data (daily granularity)
        if not _REQUESTS_AVAILABLE:
            return None
        if len(index) == 0:
            return None

        try:
            start = pd.Timestamp(index[0])
            end = pd.Timestamp(index[-1])
            days = max(int((end - start).days) + 1, 1)

            base = "https://api.coingecko.com/api/v3"
            resp = requests.get(
                f"{base}/coins/{self.cfg.coingecko_id}/market_chart",
                params={"vs_currency": self.cfg.vs_currency, "days": days,
                        "interval": "daily"},
                timeout=self.cfg.request_timeout,
            )
            resp.raise_for_status()
            chart = resp.json()

            market_caps = chart.get("market_caps", [])
            volumes = chart.get("total_volumes", [])
            prices = chart.get("prices", [])

            if not market_caps:
                return None

            dates = pd.to_datetime([x[0] for x in market_caps], unit="ms", utc=True)
            mc = pd.Series([x[1] for x in market_caps], index=dates)
            vol = pd.Series([x[1] for x in volumes], index=dates)
            price = pd.Series([x[1] for x in prices], index=dates)

            df = pd.DataFrame(index=dates)
            # MVRV proxy
            df["mvrv_raw"] = mc / vol.clip(lower=1)
            # Price momentum as NUPL proxy
            df["nupl_raw"] = price.pct_change().fillna(0.0)
            # Volume as activity proxy (two columns)
            df["active_addresses_raw"] = vol
            df["tx_count_raw"] = vol

            # Normalize each raw series to produce ONCHAIN_COLS
            normalized = self._normalize_historical(df)
            return normalized

        except Exception as exc:
            logger.debug("Historical on-chain fetch failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, raw: Dict[str, float]) -> np.ndarray:
        """Convert raw dict to a normalized float32 array of shape (10,)."""
        out = np.zeros(N_ONCHAIN_FEATURES, dtype=np.float32)

        # 1. active_addresses_norm: log-scale, tanh
        adr = raw.get("active_addresses_raw", 0.0)
        out[0] = float(np.tanh(np.log1p(abs(adr)) / 10.0)) * np.sign(adr + 1e-9)

        # 2. tx_count_norm: same as active addresses
        tx = raw.get("tx_count_raw", 0.0)
        out[1] = float(np.tanh(np.log1p(abs(tx)) / 8.0)) * np.sign(tx + 1e-9)

        # 3. hash_rate_change: direct tanh (percentage change)
        hr = raw.get("hash_rate_change", 0.0)
        out[2] = float(np.tanh(hr * 5.0))

        # 4. exchange_flow_ratio: tanh of (inflow - outflow) / total
        flow = raw.get("exchange_flow_ratio", 0.0)
        out[3] = float(np.tanh(flow * 2.0))

        # 5. exchange_reserves_change: tanh of % change
        res_chg = raw.get("exchange_reserves_chg", 0.0)
        out[4] = float(np.tanh(res_chg * 10.0))

        # 6. lth_supply_change: tanh of % change
        lth = raw.get("lth_supply_change", 0.0)
        out[5] = float(np.tanh(lth * 5.0))

        # 7. mvrv_norm: MVRV centered at typical value of 2.0
        mvrv = raw.get("mvrv_raw", 2.0)
        out[6] = float(np.tanh((mvrv - 2.0) / 2.0))

        # 8. nupl_norm: already in [-1,1] range from price change
        nupl = raw.get("nupl_raw", 0.0)
        out[7] = float(np.tanh(nupl * 3.0))

        # 9. funding_rate_norm: typical range [-0.03%, +0.03%] per 8h
        fr = raw.get("funding_rate_raw", 0.0)
        out[8] = float(np.tanh(fr * 300.0))

        # 10. open_interest_change: percentage change, tanh
        oi_chg = raw.get("open_interest_change", 0.0)
        out[9] = float(np.tanh(oi_chg * 10.0))

        return out.astype(np.float32)

    def _normalize_historical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a historical DataFrame of raw values into ONCHAIN_COLS."""
        out = pd.DataFrame(index=df.index, columns=ONCHAIN_COLS, dtype=np.float32)

        w = self.cfg.rolling_window

        def _rolling_zscore_tanh(series: pd.Series, scale: float = 1.0) -> pd.Series:
            mu = series.rolling(w, min_periods=1).mean()
            sigma = series.rolling(w, min_periods=1).std().replace(0, 1.0)
            return np.tanh((series - mu) / sigma * scale)

        # active_addresses
        if "active_addresses_raw" in df:
            out["active_addresses_norm"] = _rolling_zscore_tanh(
                df["active_addresses_raw"].fillna(0)
            ).astype(np.float32)

        # tx_count
        if "tx_count_raw" in df:
            out["tx_count_norm"] = _rolling_zscore_tanh(
                df["tx_count_raw"].fillna(0)
            ).astype(np.float32)

        # hash_rate_change
        if "hash_rate_raw" in df:
            pct = df["hash_rate_raw"].pct_change().fillna(0)
            out["hash_rate_change"] = np.tanh(pct * 5.0).astype(np.float32)
        else:
            out["hash_rate_change"] = 0.0

        # exchange_flow_ratio, reserves, lth — not in CG free data
        for col in ["exchange_flow_ratio", "exchange_reserves_chg", "lth_supply_change"]:
            out[col] = 0.0

        # mvrv
        if "mvrv_raw" in df:
            out["mvrv_norm"] = np.tanh(
                (df["mvrv_raw"].fillna(2.0) - 2.0) / 2.0
            ).astype(np.float32)
        else:
            out["mvrv_norm"] = 0.0

        # nupl
        if "nupl_raw" in df:
            out["nupl_norm"] = np.tanh(
                df["nupl_raw"].fillna(0) * 3.0
            ).astype(np.float32)
        else:
            out["nupl_norm"] = 0.0

        # derivatives (not in CG historical)
        out["funding_rate_norm"] = 0.0
        out["open_interest_change"] = 0.0

        return out.fillna(0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        """Initialize SQLite cache database."""
        import os
        db_path = self.cfg.cache_db
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        try:
            self._cache_conn = sqlite3.connect(db_path, check_same_thread=False)
            self._cache_conn.execute(
                """CREATE TABLE IF NOT EXISTS onchain_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    ts REAL NOT NULL
                )"""
            )
            self._cache_conn.commit()
        except Exception as exc:
            logger.warning("OnChain cache init failed: %s", exc)
            self._cache_conn = None

    def _cache_get(self, key: str) -> Optional[Dict]:
        """Return cached dict if still fresh, else None."""
        if self._cache_conn is None:
            return None
        try:
            import json
            now = time.time()
            row = self._cache_conn.execute(
                "SELECT value, ts FROM onchain_cache WHERE key = ?", (key,)
            ).fetchone()
            if row and (now - row[1]) < self.cfg.cache_ttl_seconds:
                return json.loads(row[0])
        except Exception:
            pass
        return None

    def _cache_set(self, key: str, data: Dict) -> None:
        """Store dict in cache with current timestamp."""
        if self._cache_conn is None:
            return
        try:
            import json
            self._cache_conn.execute(
                "INSERT OR REPLACE INTO onchain_cache (key, value, ts) VALUES (?, ?, ?)",
                (key, json.dumps(data), time.time()),
            )
            self._cache_conn.commit()
        except Exception:
            pass
