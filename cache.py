"""
cache.py — Cache TTL em memória para chamadas yfinance/Tavily custosas.

Evita pedidos duplicados quando /analisar e o scan automático apanham
o mesmo ticker em menos de TTL_MINUTES minutos.

Uso:
    from cache import get_cached, set_cached, invalidate

    data = get_cached("fundamentals", "AAPL")
    if data is None:
        data = get_fundamentals("AAPL")
        set_cached("fundamentals", "AAPL", data)

Buckets disponíveis (TTL independente):
    fundamentals   — 15 min (dados estacionários durante sessão)
    news           — 30 min (notícias mudam menos que preços)
    catalyst       — 60 min (catalisador Tavily é caro em API calls)
    rsi            — 10 min (técnico muda mais depressa)
    earnings_days  — 60 min (só muda uma vez por dia na prática)
"""

import time
import threading
import logging
from typing import Any

_TTL: dict[str, int] = {
    "fundamentals":  15 * 60,
    "news":          30 * 60,
    "catalyst":      60 * 60,
    "rsi":           10 * 60,
    "earnings_days": 60 * 60,
}

# _store[bucket][key] = (value, timestamp)
_store: dict[str, dict[str, tuple[Any, float]]] = {b: {} for b in _TTL}
_lock  = threading.Lock()


def get_cached(bucket: str, key: str) -> Any | None:
    """
    Devolve o valor em cache se existir e não tiver expirado.
    Devolve None caso contrário.
    """
    ttl = _TTL.get(bucket)
    if ttl is None:
        return None
    with _lock:
        entry = _store[bucket].get(key)
    if entry is None:
        return None
    value, ts = entry
    if time.monotonic() - ts > ttl:
        with _lock:
            _store[bucket].pop(key, None)
        return None
    return value


def set_cached(bucket: str, key: str, value: Any) -> None:
    """Guarda valor em cache com timestamp actual."""
    if bucket not in _store:
        logging.warning(f"[cache] bucket desconhecido: {bucket}")
        return
    with _lock:
        _store[bucket][key] = (value, time.monotonic())


def invalidate(bucket: str, key: str | None = None) -> None:
    """
    Invalida uma chave específica ou todo o bucket se key=None.
    """
    with _lock:
        if key is None:
            _store[bucket].clear()
        else:
            _store[bucket].pop(key, None)


def cache_stats() -> str:
    """Linha de debug com contagem de entradas por bucket."""
    now = time.monotonic()
    parts = []
    with _lock:
        for bucket, entries in _store.items():
            ttl = _TTL[bucket]
            live = sum(1 for _, (_, ts) in entries.items() if now - ts <= ttl)
            parts.append(f"{bucket}: {live}/{len(entries)}")
    return "Cache — " + " | ".join(parts)
