import os
import time
import logging
from trigger_client import get_losers, get_profile
from filtering import passes_fast_filters, rank_candidates

DROP_THRESHOLD = float(os.environ.get('DROP_THRESHOLD', '10'))
MIN_MARKET_CAP = int(os.environ.get('MIN_MARKET_CAP', '10000000000'))
MIN_PRICE = float(os.environ.get('MIN_PRICE', '5'))
MAX_DEEP_ANALYSIS_PER_RUN = int(os.environ.get('MAX_DEEP_ANALYSIS_PER_RUN', '3'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def run_scan() -> list[dict]:
    losers = get_losers()
    if not losers:
        logging.info('Sem losers retornados pela FMP.')
        return []

    losers = [x for x in losers if x.get('change_pct', 0) <= -DROP_THRESHOLD]
    logging.info(f'{len(losers)} com queda >= {DROP_THRESHOLD}%')

    filtered = []
    for stock in losers:
        symbol = stock['symbol']
        try:
            profile = get_profile(symbol)
            ok, reason = passes_fast_filters(stock, profile, MIN_MARKET_CAP, MIN_PRICE)
            if not ok:
                logging.info(f'{symbol}: rejeitada - {reason}')
                continue

            stock['market_cap'] = float(profile.get('mktCap') or profile.get('marketCap') or 0)
            stock['sector'] = profile.get('sector') or ''
            stock['company_name'] = profile.get('companyName') or stock.get('name') or symbol
            filtered.append(stock)
            logging.info(f"{symbol}: passou filtro | queda {stock['change_pct']:.1f}% | cap {stock['market_cap']/1e9:.1f}B")
            time.sleep(0.4)
        except Exception as e:
            logging.error(f'{symbol}: erro no profile - {e}')
            continue

    ranked = rank_candidates(filtered)
    finalists = ranked[:MAX_DEEP_ANALYSIS_PER_RUN]
    logging.info(f'Finalistas: {len(finalists)}')

    for item in finalists:
        logging.info(
            f"FINALISTA {item['symbol']} | {item['change_pct']:.1f}% | "
            f"cap {item['market_cap']/1e9:.1f}B | sector {item.get('sector','')}"
        )

    return finalists


if __name__ == '__main__':
    logging.info('=' * 50)
    logging.info('Stock Alert Bot V2 iniciado')
    logging.info(f'Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.1f}B | Min price: ${MIN_PRICE:.1f}')
    logging.info('=' * 50)
    run_scan()
