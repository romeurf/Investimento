import os
import time
import logging
import requests

FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
BASE_URL = 'https://financialmodelingprep.com/api/v3'


def _get(endpoint: str, params: dict | None = None):
    query = {'apikey': FMP_API_KEY}
    if params:
        query.update(params)
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, params=query, timeout=30)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get('Error Message'):
        raise ValueError(data['Error Message'])
    return data


def normalize_change_pct(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip().replace('(', '').replace(')', '').replace('%', '').replace(',', '.')
        return float(txt)
    return 0.0


def get_losers() -> list[dict]:
    data = _get('stock_market/losers')
    out = []
    for item in data or []:
        symbol = item.get('symbol') or item.get('ticker') or ''
        if not symbol:
            continue
        out.append({
            'symbol': symbol.upper(),
            'name': item.get('name') or item.get('companyName') or symbol.upper(),
            'price': item.get('price') or 0,
            'change_pct': normalize_change_pct(item.get('changesPercentage', 0)),
        })
    logging.info(f'FMP losers brutos: {len(out)}')
    return out


def get_profile(symbol: str) -> dict:
    data = _get(f'profile/{symbol}')
    if not data:
        return {}
    return data[0] if isinstance(data, list) and data else {}
