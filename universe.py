"""
universe.py — Universo global de tickers para o DipRadar 2.0.

Fontes:
  - S&P 500       : Wikipedia (scraped dinamicamente)
  - Nasdaq 100    : Wikipedia (scraped dinamicamente)
  - STOXX 600     : Lista estática curada (top 200 por liquidez)
  - FTSE 100      : Lista estática curada
  - Carteira atual: Hardcoded (tickers do utilizador)

Uso:
  from universe import get_full_universe
  tickers = get_full_universe()   # list[str], ~1200 tickers únicos
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Optional

# ── Carteira actual do utilizador ────────────────────────────────────────────
# Estes tickers entram sempre no universo independentemente dos índices.
# Actualiza esta lista com /buy e /sell (gerido pelo portfolio.py).
USER_PORTFOLIO: list[str] = [
    "NVO", "ADBE", "UBER", "MSFT", "PINS",
    "ADP", "CRM", "VICI", "CRWD", "PLTR", "NOW", "DUOL",
]

# Watchlist pessoal (tickers de interesse, mesmo fora da carteira)
USER_WATCHLIST: list[str] = [
    "O", "MDT", "ABBV", "LMT", "RTX",
    "PANW", "TSM", "AVGO", "ALV.DE", "IEMA",
]

# ── STOXX Europe 600 — top 200 por liquidez/relevância ───────────────────────
_STOXX200: list[str] = [
    # Alemanha
    "SAP.DE", "SIE.DE", "ALV.DE", "MRK.DE", "BMW.DE", "MBG.DE",
    "BAS.DE", "BAYN.DE", "DTE.DE", "DBK.DE", "VOW3.DE", "ADS.DE",
    "HNR1.DE", "RWE.DE", "HEI.DE", "ENR.DE", "EOAN.DE", "CON.DE",
    "FRE.DE", "ZAL.DE", "AIR.DE", "LIN.DE",
    # França
    "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA", "BNP.PA",
    "ACA.PA", "SGO.PA", "SU.PA", "DG.PA", "RI.PA", "CS.PA",
    "KER.PA", "ATO.PA", "CAP.PA", "VIE.PA", "LR.PA", "DSY.PA",
    "HO.PA", "ML.PA", "ORA.PA",
    # Holanda
    "ASML.AS", "HEIA.AS", "PHIA.AS", "REN.AS", "UNA.AS",
    "INGA.AS", "ABN.AS", "NN.AS", "RAND.AS", "AKZA.AS",
    # Reino Unido
    "SHEL.L", "AZN.L", "HSBA.L", "BP.L", "RIO.L", "GSK.L",
    "ULVR.L", "DGE.L", "BATS.L", "LLOY.L", "BARC.L", "VOD.L",
    "BT-A.L", "REL.L", "NG.L", "SSE.L", "IMB.L", "EXPN.L",
    "ABF.L", "FERG.L", "CRH.L", "LSEG.L", "NWG.L", "PRU.L",
    # Suíça
    "NESN.SW", "NOVN.SW", "ROG.SW", "UBSG.SW", "CSGN.SW",
    "ABBN.SW", "ZURN.SW", "GIVN.SW", "LONN.SW", "CFR.SW",
    # Espanha
    "ITX.MC", "SAN.MC", "TEF.MC", "BBVA.MC", "IBE.MC",
    "AMS.MC", "REP.MC", "CABK.MC",
    # Itália
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "STM.MI",
    "LDO.MI", "MB.MI", "FCA.MI",
    # Suécia
    "VOLV-B.ST", "ERIC-B.ST", "ATCO-A.ST", "SEB-A.ST",
    "SHB-A.ST", "SWED-A.ST", "INVE-B.ST",
    # Dinamarca
    "NOVO-B.CO", "ORSTED.CO", "CARL-B.CO", "DSV.CO",
    # Noruega
    "EQNR.OL", "DNB.OL", "MOWI.OL",
    # Finlândia
    "NOKIA.HE", "FORTUM.HE", "NESTE.HE",
    # Bélgica
    "UCB.BR", "ABI.BR", "SOLB.BR",
    # Irlanda
    "CRH.I", "KYGA.I",
    # Portugal
    "EDP.LS", "GALP.LS", "BCP.LS",
    # Áustria
    "VIG.VI", "OMV.VI",
    # Polónia
    "PKN.WA", "PKO.WA",
]

# ── FTSE 100 ─────────────────────────────────────────────────────────────────
_FTSE100: list[str] = [
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "RIO.L",
    "GSK.L", "DGE.L", "REL.L", "BATS.L", "LLOY.L", "BARC.L",
    "VOD.L", "BT-A.L", "NG.L", "SSE.L", "IMB.L", "CRH.L",
    "LSEG.L", "NWG.L", "PRU.L", "EXPN.L", "ABF.L", "FERG.L",
    "TSCO.L", "CPG.L", "RKT.L", "WPP.L", "IAG.L", "EZJ.L",
    "AAL.L", "ANTO.L", "BHP.L", "GLEN.L", "EVR.L", "MNDI.L",
    "SMDS.L", "SMT.L", "III.L", "LAND.L", "SGRO.L", "HMSO.L",
    "BLND.L", "DLN.L", "PSN.L", "BWY.L", "TW.L", "BA.L",
    "RR.L", "QQ.L", "AUTO.L", "OCDO.L", "MKS.L", "NEXT.L",
    "JD.L", "SPX.L", "RS1.L", "DCC.L", "SKG.L", "IMI.L",
    "PHNX.L", "LGEN.L", "AV.L", "ADM.L", "HSBA.L", "STAN.L",
    "INVP.L", "MNDI.L", "RTO.L", "SBRY.L", "MRO.L",
]


def _fetch_sp500() -> list[str]:
    """Scrape S&P 500 da Wikipedia. Fallback para lista estática se falhar."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={"id": "constituents"})
        tickers = tables[0]["Symbol"].tolist()
        # Yahoo Finance usa - em vez de . em alguns tickers (ex: BRK.B → BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        logging.info(f"[universe] S&P 500: {len(tickers)} tickers obtidos da Wikipedia")
        return tickers
    except Exception as e:
        logging.warning(f"[universe] Falha ao obter S&P 500 da Wikipedia: {e}. A usar fallback.")
        return _SP500_FALLBACK


def _fetch_nasdaq100() -> list[str]:
    """Scrape Nasdaq 100 da Wikipedia. Fallback para lista estática se falhar."""
    try:
        import pandas as pd
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        tables = pd.read_html(url)
        # Encontra a tabela com coluna 'Ticker'
        for t in tables:
            if "Ticker" in t.columns:
                tickers = t["Ticker"].tolist()
                logging.info(f"[universe] Nasdaq 100: {len(tickers)} tickers obtidos da Wikipedia")
                return [str(t) for t in tickers]
        raise ValueError("Tabela Nasdaq 100 não encontrada")
    except Exception as e:
        logging.warning(f"[universe] Falha ao obter Nasdaq 100 da Wikipedia: {e}. A usar fallback.")
        return _NASDAQ100_FALLBACK


@lru_cache(maxsize=1)
def get_full_universe(refresh: bool = False) -> list[str]:
    """
    Devolve lista única de tickers do universo completo.
    Cache de sessão — chama get_full_universe.cache_clear() para forçar refresh.

    Ordem de prioridade (deduplicação mantém primeira ocorrência):
      1. Carteira do utilizador (sempre monitorizada)
      2. Watchlist pessoal
      3. S&P 500
      4. Nasdaq 100
      5. STOXX Europe 200
      6. FTSE 100
    """
    sp500    = _fetch_sp500()
    ndx100   = _fetch_nasdaq100()

    all_tickers = (
        USER_PORTFOLIO
        + USER_WATCHLIST
        + sp500
        + ndx100
        + _STOXX200
        + _FTSE100
    )

    # Deduplica preservando ordem
    seen:    set[str] = set()
    unique:  list[str] = []
    for t in all_tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)

    logging.info(f"[universe] Universo total: {len(unique)} tickers únicos")
    return unique


def get_universe_stats() -> dict:
    """Estatísticas do universo para debug/logging."""
    sp500  = _fetch_sp500()
    ndx100 = _fetch_nasdaq100()
    full   = get_full_universe()
    return {
        "total":       len(full),
        "sp500":       len(sp500),
        "nasdaq100":   len(ndx100),
        "stoxx200":    len(_STOXX200),
        "ftse100":     len(_FTSE100),
        "portfolio":   len(USER_PORTFOLIO),
        "watchlist":   len(USER_WATCHLIST),
    }


# ── Fallbacks estáticos (usados se Wikipedia falhar) ─────────────────────────
# Top 50 de cada índice para garantir cobertura mínima

_SP500_FALLBACK: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "JPM", "TSLA", "UNH", "V", "XOM", "MA", "JNJ", "PG",
    "HD", "COST", "ABBV", "MRK", "CVX", "WMT", "NFLX", "BAC",
    "CRM", "AMD", "KO", "PEP", "TMO", "ACN", "LIN", "MCD", "CSCO",
    "ABT", "TXN", "ORCL", "PM", "NKE", "ADBE", "DHR", "WFC", "GE",
    "NEE", "RTX", "BMY", "UPS", "AMGN", "QCOM", "LOW", "SPGI",
    "INTU", "HON", "CAT", "DE", "GS", "AXP", "SBUX", "PLD",
    "ISRG", "GILD", "BKNG", "MDT", "VRTX", "ADI", "REGN", "NOW",
    "PANW", "CRWD", "PLTR", "DUOL", "PINS", "ADP", "VICI", "O",
]

_NASDAQ100_FALLBACK: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",
    "COST", "NFLX", "ASML", "AMD", "ADBE", "QCOM", "INTU", "TXN",
    "AMAT", "CMCSA", "CSCO", "PEP", "HON", "ISRG", "BKNG", "SBUX",
    "REGN", "VRTX", "PANW", "CRWD", "MRVL", "GILD", "MDLZ", "MU",
    "ADI", "LRCX", "KDP", "MAR", "KLAC", "SNPS", "CDNS", "ORLY",
    "AZN", "CTAS", "PYPL", "MELI", "MNST", "FTNT", "NXPI", "PAYX",
    "ABNB", "DXCM",
]
