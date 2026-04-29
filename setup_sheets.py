"""
setup_sheets.py — Script utilitário one-time para inicializar a folha Google Sheets.

Execução (local, uma única vez):
  export GOOGLE_SHEETS_CREDENTIALS='{ ...conteúdo do JSON da Service Account... }'
  export GOOGLE_SHEET_ID='1AbCdEfGhIjKlMnOpQrStUvWxYz...'
  python setup_sheets.py

O script cria (ou recria) as abas 'Liquidez' e 'Posicoes' com cabeçalhos
na folha identificada por GOOGLE_SHEET_ID.

NÃO apaga dados existentes sem confirmação.
"""

import json
import os
import sys

CREDENTIALS_JSON = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "")
SHEET_ID         = os.getenv("GOOGLE_SHEET_ID", "")

if not CREDENTIALS_JSON or not SHEET_ID:
    print("❌ Faltam variáveis de ambiente:")
    print("   GOOGLE_SHEETS_CREDENTIALS — conteúdo JSON da Service Account")
    print("   GOOGLE_SHEET_ID           — ID da folha (URL entre /d/ e /edit)")
    sys.exit(1)

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    print("❌ Dependências em falta. Corre: pip install gspread google-auth")
    sys.exit(1)


def main() -> None:
    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds  = Credentials.from_service_account_info(json.loads(CREDENTIALS_JSON), scopes=scopes)
    client = gspread.authorize(creds)

    try:
        sheet = client.open_by_key(SHEET_ID)
        print(f"✅ Folha encontrada: '{sheet.title}'")
    except Exception as e:
        print(f"❌ Não foi possível abrir a folha com ID '{SHEET_ID}': {e}")
        sys.exit(1)

    existing_titles = [ws.title for ws in sheet.worksheets()]

    # ── Aba Liquidez ──────────────────────────────────────────────────────
    if "Liquidez" in existing_titles:
        ws_liq = sheet.worksheet("Liquidez")
        current = ws_liq.acell("A2").value
        if current:
            print(f"⚠️  Aba 'Liquidez' já existe com valor: {current}€")
            resp = input("   Reinicializar a zero? (s/N): ").strip().lower()
            if resp == "s":
                ws_liq.update([["Liquidez"]], "A1")
                ws_liq.update([[0.0]], "A2")
                print("   ✅ Liquidez reinicializada a 0.0")
            else:
                print("   ↩ Aba 'Liquidez' mantida sem alterações.")
        else:
            ws_liq.update([["Liquidez"]], "A1")
            ws_liq.update([[0.0]], "A2")
            print("✅ Aba 'Liquidez' inicializada")
    else:
        ws_liq = sheet.add_worksheet(title="Liquidez", rows=10, cols=2)
        ws_liq.update([["Liquidez"]], "A1")
        ws_liq.update([[0.0]], "A2")
        print("✅ Aba 'Liquidez' criada")

    # ── Aba Posicoes ──────────────────────────────────────────────────────
    HEADERS = [
        "Ticker", "Entry_Date", "Entry_Price", "Quantity",
        "Entry_Category", "Entry_Score", "Last_Price",
        "Last_Score", "Last_Update", "Degradation_Alerted",
    ]

    if "Posicoes" in existing_titles:
        ws_pos = sheet.worksheet("Posicoes")
        all_vals = ws_pos.get_all_values()
        n_rows   = len([r for r in all_vals if any(c.strip() for c in r)])
        if n_rows > 1:
            print(f"⚠️  Aba 'Posicoes' já existe com {n_rows - 1} posição(ões).")
            resp = input("   Apagar tudo e reinicializar? (s/N): ").strip().lower()
            if resp == "s":
                ws_pos.clear()
                ws_pos.append_row(HEADERS)
                print("   ✅ Aba 'Posicoes' reinicializada")
            else:
                print("   ↩ Aba 'Posicoes' mantida sem alterações.")
        else:
            ws_pos.clear()
            ws_pos.append_row(HEADERS)
            print("✅ Aba 'Posicoes' inicializada com cabeçalhos")
    else:
        ws_pos = sheet.add_worksheet(title="Posicoes", rows=200, cols=len(HEADERS))
        ws_pos.append_row(HEADERS)
        print("✅ Aba 'Posicoes' criada com cabeçalhos")

    print()
    print("🚀 Setup completo! Variáveis a adicionar no Railway:")
    print(f"   GOOGLE_SHEET_ID           = {SHEET_ID}")
    print( "   GOOGLE_SHEETS_CREDENTIALS = <conteúdo do JSON da Service Account>")
    print()
    print("📌 Próximo passo: corre /admin_backfill_ml no Telegram para popular a watchlist.")


if __name__ == "__main__":
    main()
