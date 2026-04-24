
def is_common_stock_symbol(symbol: str) -> bool:
    if not symbol:
        return False
    banned_tokens = ['W', 'U', 'R']
    if any(ch in symbol for ch in ['^', '/', ' ']):
        return False
    if symbol.endswith(tuple(['.WS', '-WS', '.WT', '-WT', '.U', '-U', '.R', '-R'])):
        return False
    if len(symbol) > 5 and symbol[-1] in banned_tokens:
        return False
    return True


def passes_fast_filters(stock: dict, profile: dict, min_market_cap: int, min_price: float) -> tuple[bool, str]:
    price = float(stock.get('price') or profile.get('price') or 0)
    market_cap = float(profile.get('mktCap') or profile.get('marketCap') or 0)
    is_etf = bool(profile.get('isEtf'))
    is_fund = bool(profile.get('isFund'))
    exchange = (profile.get('exchangeShortName') or profile.get('exchange') or '').upper()

    if not is_common_stock_symbol(stock.get('symbol', '')):
        return False, 'símbolo inválido'
    if is_etf or is_fund:
        return False, 'ETF/fund'
    if exchange and exchange not in {'NASDAQ', 'NYSE', 'AMEX'}:
        return False, f'exchange {exchange}'
    if price < min_price:
        return False, f'preço {price:.2f} < {min_price:.2f}'
    if market_cap < min_market_cap:
        return False, f'market cap {market_cap/1e9:.1f}B < {min_market_cap/1e9:.1f}B'
    return True, 'ok'


def rank_candidates(candidates: list[dict]) -> list[dict]:
    return sorted(
        candidates,
        key=lambda x: (
            x.get('change_pct', 0),
            -(x.get('market_cap', 0) or 0)
        )
    )
