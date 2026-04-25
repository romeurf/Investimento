"""
Stock Alert Bot — monitoriza quedas fortes e analisa fundamentos.
Trigger: EODHD Screener
Fundamentos/notícias: Financial Modeling Prep
"""

import os
import time
import logging
import schedule
import requests
from datetime import datetime

from market_client import screen_big_drops, get_fundamentals, get_news, get_historical_pe
from sectors import get_sector_config, score_fundamentals
from valuation import format_valuation_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
EODHD_API_KEY = os.environ.get("EODHD_API_KEY", "")
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

DROP_THRESHOLD = float(os.environ.get("DROP_THRESHOLD", "10"))
MIN_MARKET_CAP = int(os.environ.get("MIN_MARKET_CAP", "10000000000"))
SCAN_EVERY_MINUTES = int(os.environ.get("SCAN_EVERY_MINUTES", "30"))

_alerted_today: set[str] = set()


def validate_env() -> None:
    missing = []
    if not EODHD_API_KEY:
        missing.append("EODHD_API_KEY")
    if not FMP_API_KEY:
        logging.warning("FMP_API_KEY ausente — o trigger funciona, mas fundamentos/notícias podem falhar.")
    if missing:
        raise RuntimeError(f"Faltam variáveis de ambiente obrigatórias: {', '.join(missing)}")


def send_telegram(message: str, parse_mode: str = "Markdown") -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return True

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4096],
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Telegram error: {e}")
        return False


def build_alert(stock: dict, fundamentals: dict, historical_pe: float | None,
                news: list, verdict: str, emoji: str, reasons: list) -> str:
    sector = fundamentals.get("sector", "")
    sector_cfg = get_sector_config(sector)
    name = fundamentals.get("name", stock.get("name", stock["symbol"]))
    change = stock.get("change_pct", 0)
    price = fundamentals.get("price") or stock.get("price", "N/D")
    mc_b = (fundamentals.get("market_cap") or stock.get("market_cap") or 0) / 1e9
    source = stock.get("source", "market screener")

    header = (
        f"📉 *{stock['symbol']} — {name}*\n"
        f"Queda: *{change:.1f}%* hoje | Preço: ${price} | Cap: ${mc_b:.1f}B\n"
        f"Sector: {sector_cfg.get('label', sector)}\n"
        f"Fonte trigger: {source}\n"
    )

    verdict_block = f"\n*Veredito: {emoji} {verdict}*\n"
    if reasons:
        verdict_block += "\n".join(f"_{r}_" for r in reasons) + "\n"

    valuation_block = "\n*📊 Fundamentos:*\n"
    valuation_block += format_valuation_block(fundamentals, historical_pe, sector)

    news_block = "\n\n*📰 Notícias:*\n"
    if news:
        for item in news[:3]:
            title = item.get("title", "")[:70]
            url = item.get("url", "")
            src = item.get("source", "")
            if url:
                news_block += f" • [{title}]({url})" + (f" _{src}_" if src else "") + "\n"
            else:
                news_block += f" • {title}" + (f" _{src}_" if src else "") + "\n"
    else:
        news_block += " • Sem notícias recentes\n"

    footer = f"\n_⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}_"
    return header + verdict_block + valuation_block + news_block + footer


def run_scan() -> None:
    today = datetime.now().date().isoformat()
    logging.info(f"A correr scan — {datetime.now().strftime('%H:%M')}")

    losers = screen_big_drops(
        min_drop_pct=DROP_THRESHOLD,
        min_market_cap=MIN_MARKET_CAP,
    )

    if not losers:
        logging.info("Nenhuma acção com queda suficiente, ou trigger indisponível.")
        return

    logging.info(f"{len(losers)} acção(ões) com queda ≥{DROP_THRESHOLD}%")

    alerts_sent = 0
    for stock in losers:
        symbol = stock.get("symbol")
        if not symbol:
            continue

        alert_key = f"{symbol}_{today}"
        if alert_key in _alerted_today:
            continue

        try:
            logging.info(f"A analisar {symbol} ({stock.get('change_pct', 0):.1f}%)...")

            fundamentals = get_fundamentals(symbol)
            sector = fundamentals.get("sector", "")
            verdict, emoji, reasons = score_fundamentals(fundamentals, sector)

            if verdict == "EVITAR":
                logging.info(f"{symbol}: EVITAR — a saltar")
                _alerted_today.add(alert_key)
                continue

            historical_pe = get_historical_pe(symbol)
            news = get_news(symbol)
            message = build_alert(stock, fundamentals, historical_pe, news, verdict, emoji, reasons)
            success = send_telegram(message)

            if success:
                _alerted_today.add(alert_key)
                alerts_sent += 1
                logging.info(f"✅ Alerta enviado: {symbol} ({verdict})")

            time.sleep(1.5)

        except Exception as e:
            logging.error(f"Erro a processar {symbol}: {e}")
            continue

    if alerts_sent == 0 and losers:
        logging.info("Acções com queda encontradas mas fundamentos fracos — sem alertas.")


def send_daily_summary() -> None:
    losers = screen_big_drops(min_drop_pct=3.0, min_market_cap=MIN_MARKET_CAP)
    if not losers:
        logging.info("Resumo diário sem dados.")
        return

    lines = [f"*📋 Resumo — {datetime.now().strftime('%d/%m/%Y %H:%M')}*\n"]
    lines.append("Acções com queda ≥3% hoje:\n")

    for s in losers[:10]:
        sym = s.get("symbol", "?")
        chg = s.get("change_pct", 0)
        mc_b = (s.get("market_cap") or 0) / 1e9
        lines.append(f"📉 *{sym}*: {chg:.1f}% (cap ${mc_b:.1f}B)")

    send_telegram("\n".join(lines))


def clear_alert_cache() -> None:
    _alerted_today.clear()
    logging.info("Cache diária de alertas limpa.")


if __name__ == "__main__":
    validate_env()

    logging.info("=" * 60)
    logging.info("Stock Alert Bot iniciado")
    logging.info(f"Trigger: EODHD Screener | Threshold: {DROP_THRESHOLD}% | Min cap: ${MIN_MARKET_CAP/1e9:.1f}B")
    logging.info(f"Scan a cada {SCAN_EVERY_MINUTES} minutos")
    logging.info("=" * 60)

    send_telegram(
        f"🤖 *Bot iniciado*\n"
        f"Trigger: EODHD Screener\n"
        f"Threshold: ≥{DROP_THRESHOLD}% de queda\n"
        f"Min market cap: ${MIN_MARKET_CAP/1e9:.1f}B\n"
        f"_Scan a cada {SCAN_EVERY_MINUTES} minutos_"
    )

    schedule.every(SCAN_EVERY_MINUTES).minutes.do(run_scan)
    schedule.every().day.at("18:00").do(send_daily_summary)
    schedule.every().day.at("00:01").do(clear_alert_cache)

    run_scan()

    while True:
        schedule.run_pending()
        time.sleep(60)
