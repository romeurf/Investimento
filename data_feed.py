"""
data_feed.py — Módulo de ingestão de dados EOD para o DipRadar 2.0.

Fonte primária : Tiingo API (EOD limpos, sem rate limit agressivo)
Fallback       : yfinance (fundamentais e tickers não cobertos pela Tiingo)

Variáveis de ambiente necessárias (Railway):
  TIINGO_API_KEY   — chave da API Tiingo (gratuita em api.tiingo.com)

Auto-Recovery:
  Todos os pedidos têm try/except defensivo.
  Tickers que falham devolvem DataFrame vazio — NUNCA crasham o scan.
  get_bulk_eod() devolve também a lista de tickers falhados para telemetria.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────

TIINGO_API_KEY: str = os.getenv("TIINGO_API_KEY", "")
TIINGO_BASE    = "https://api.tiingo.com/tiingo/daily"
TIINGO_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {TIINGO_API_KEY}",
}

_TIINGO_EXCHANGE_MAP: dict[str, str] = {
    ".DE": "-DE", ".PA": "-PA", ".AS": "-AS", ".MC": "-MC",
    ".MI": "-MI", ".L":  "-L",  ".SW": "-SW", ".ST": "-ST",
    ".CO": "-CO", ".OL": "-OL", ".HE": "-HE", ".BR": "-BR",
    ".LS": "-LS", ".VI": "-VI", ".WA": "-WA", ".I":  "-I",
}

TIINGO_UNSUPPORTED: set[str] = set()

# Contadores de telemetria desta sessão
_session_failed_tickers: list[str] = []


# ─────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────

def _to_tiingo_ticker(ticker: str) -> str:
    t = ticker.upper()
    for yf_suffix, tiingo_suffix in _TIINGO_EXCHANGE_MAP.items():
        if t.endswith(yf_suffix):
            return t[: -len(yf_suffix)] + tiingo_suffix
    return t


def _date_range(lookback_days: int) -> tuple[str, str]:
    end   = date.today()
    start = end - timedelta(days=lookback_days)
    return start.isoformat(), end.isoformat()


def _parse_tiingo_response(data: list[dict], ticker: str) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    try:
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)
        rename = {
            "adjClose":  "Adj Close",
            "adjOpen":   "Open",
            "adjHigh":   "High",
            "adjLow":    "Low",
            "adjVolume": "Volume",
            "close":     "Close",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df["ticker"] = ticker
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        cols = [c for c in ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"] if c in df.columns]
        return df[cols]
    except Exception as e:
        log.warning(f"[data_feed] parse Tiingo erro para {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────
# Fontes internas — blindadas
# ─────────────────────────────────────────────────────────────────────────

def _tiingo_fetch(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch Tiingo REST API — nunca crasha, devolve DataFrame vazio em falha."""
    try:
        tiingo_ticker        = _to_tiingo_ticker(ticker)
        start_date, end_date = _date_range(lookback_days)
        url    = f"{TIINGO_BASE}/{tiingo_ticker}/prices"
        params = {"startDate": start_date, "endDate": end_date, "resampleFreq": "daily"}

        r = requests.get(url, headers=TIINGO_HEADERS, params=params, timeout=10)
        if r.status_code == 404:
            log.debug(f"[data_feed] Tiingo 404 para {ticker} (ticker não coberto)")
            return pd.DataFrame()
        r.raise_for_status()
        data = r.json()
        if not data:
            return pd.DataFrame()
        return _parse_tiingo_response(data, ticker)

    except requests.exceptions.Timeout:
        log.warning(f"[data_feed] Tiingo timeout para {ticker}")
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        log.warning(f"[data_feed] Tiingo connection error para {ticker}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        log.warning(f"[data_feed] Tiingo HTTP erro para {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        log.warning(f"[data_feed] Tiingo erro inesperado para {ticker}: {e}")
        return pd.DataFrame()


def _yfinance_fetch(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fallback yfinance — nunca crasha, devolve DataFrame vazio em falha."""
    try:
        import yfinance as yf
        start_date, end_date = _date_range(lookback_days)
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            show_errors=False,
        )
        if raw is None or raw.empty:
            log.debug(f"[data_feed] yfinance devolveu dados vazios para {ticker}")
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.reset_index().rename(columns={"Date": "date", "index": "date"})
        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)

        if "Adj Close" not in raw.columns and "Close" in raw.columns:
            raw["Adj Close"] = raw["Close"]

        raw["ticker"] = ticker
        cols = [c for c in ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticker"] if c in raw.columns]
        result = raw[cols]

        # Valida que temos dados suficientes
        if len(result) < 2:
            log.debug(f"[data_feed] yfinance dados insuficientes para {ticker} ({len(result)} linhas)")
            return pd.DataFrame()

        return result

    except Exception as e:
        log.warning(f"[data_feed] yfinance erro para {ticker}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────────────────────────────────

def get_eod_prices(
    ticker: str,
    lookback_days: int = 60,
    force_yfinance: bool = False,
) -> pd.DataFrame:
    """
    Retorna DataFrame com preços EOD para um ticker.
    NUNCA lança excepção — devolve DataFrame vazio se ambas as fontes falharem.
    """
    try:
        if not force_yfinance and TIINGO_API_KEY and ticker not in TIINGO_UNSUPPORTED:
            df = _tiingo_fetch(ticker, lookback_days)
            if not df.empty:
                return df
            TIINGO_UNSUPPORTED.add(ticker)
            log.info(f"[data_feed] Tiingo sem dados para {ticker} — fallback yfinance")

        return _yfinance_fetch(ticker, lookback_days)

    except Exception as e:
        log.error(f"[data_feed] get_eod_prices falha total para {ticker}: {e}")
        return pd.DataFrame()


def get_bulk_eod(
    tickers: list[str],
    lookback_days: int = 60,
    delay_between: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Fetch EOD em bulk. Nunca crasha.

    Returns:
      Dict {ticker: DataFrame} — falhas têm DataFrame vazio.
      Acede a get_bulk_eod.failed_tickers para lista de tickers falhados.
    """
    global _session_failed_tickers
    results: dict[str, pd.DataFrame] = {}
    failed:  list[str]               = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        try:
            df = get_eod_prices(ticker, lookback_days=lookback_days)
            results[ticker] = df
            if df.empty:
                failed.append(ticker)
                log.warning(f"[data_feed] {ticker}: dados vazios (possível delisting ou falha API)")
            if i % 50 == 0:
                log.info(f"[data_feed] bulk {i}/{total} tickers processados")
        except Exception as e:
            log.error(f"[data_feed] Erro inesperado a buscar {ticker}: {e}")
            results[ticker] = pd.DataFrame()
            failed.append(ticker)

        time.sleep(delay_between)

    ok = total - len(failed)
    log.info(f"[data_feed] Bulk EOD completo: {ok} OK, {len(failed)} falhas de {total}")
    if failed:
        log.warning(f"[data_feed] Tickers falhados: {', '.join(failed)}")

    # Expõe lista de falhados para telemetria no main.py
    get_bulk_eod.failed_tickers = failed
    _session_failed_tickers     = failed
    return results


# Inicializa atributo para evitar AttributeError antes do primeiro bulk
get_bulk_eod.failed_tickers = []


def get_failed_tickers() -> list[str]:
    """Devolve lista de tickers que falharam no último bulk scan desta sessão."""
    return list(_session_failed_tickers)


def get_latest_price(ticker: str) -> Optional[float]:
    """Devolve o último preço de fecho (Adj Close). Devolve None se falhar."""
    try:
        df = get_eod_prices(ticker, lookback_days=5)
        if df.empty:
            return None
        return float(df["Adj Close"].iloc[-1])
    except Exception:
        return None


def is_tiingo_available() -> bool:
    """Verifica se a chave Tiingo está configurada e funcional."""
    if not TIINGO_API_KEY:
        return False
    try:
        url = f"{TIINGO_BASE}/AAPL/prices"
        r   = requests.get(
            url, headers=TIINGO_HEADERS,
            params={"startDate": date.today().isoformat()},
            timeout=5,
        )
        return r.status_code == 200
    except Exception:
        return False
