from __future__ import annotations

import json
import math
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
TEMPLATES_DIR = ROOT / "templates"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

DOCS_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)


FX_TICKERS = {
    "EUR": None,
    "USD": "EURUSD=X",
    "DKK": "EURDKK=X",
    "CHF": "EURCHF=X",
    "HKD": "EURHKD=X",
    "CNY": "EURCNY=X",
    "GBP": "EURGBP=X",
    "JPY": "EURJPY=X",
}


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def download_close(ticker: str, start: str) -> pd.Series:
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No Yahoo Finance data for ticker {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        else:
            close = df.xs("Close", axis=1, level=0).iloc[:, 0]
    else:
        close = df["Close"]

    close = close.dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    if close.empty:
        raise ValueError(f"No close prices available for ticker {ticker}")

    return close


def fx_to_eur_series(currency: str, start: str, index: pd.DatetimeIndex) -> pd.Series:
    if currency == "EUR":
        return pd.Series(1.0, index=index)

    fx_ticker = FX_TICKERS.get(currency)
    if not fx_ticker:
        raise ValueError(f"FX ticker missing for currency {currency}")

    # Yahoo convention: EURUSD=X = USD per 1 EUR.
    # To convert USD -> EUR, use 1 / EURUSD.
    eur_to_ccy = download_close(fx_ticker, start=start)
    ccy_to_eur = 1.0 / eur_to_ccy
    return ccy_to_eur.reindex(index, method="ffill")


def first_after_or_equal(series: pd.Series, d: pd.Timestamp):
    s = series[series.index >= d]
    if s.empty:
        return None
    return safe_float(s.iloc[0])


def aggregate_weights(rows: list[dict], key: str, total_value: float) -> dict:
    out: Dict[str, float] = {}
    for row in rows:
        label = row.get(key) or "Other"
        value = row.get("current_value_eur") or 0.0
        out[label] = out.get(label, 0.0) + value

    if total_value <= 0:
        return {}

    return dict(sorted({k: v / total_value * 100 for k, v in out.items()}.items(), key=lambda x: x[1], reverse=True))


def load_previous_snapshot() -> pd.DataFrame | None:
    path = DATA_DIR / "last_snapshot.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def save_snapshot(rows: list[dict]):
    snapshot = pd.DataFrame([
        {
            "instrument_id": r["instrument_id"],
            "ticker": r["ticker"],
            "latest_date": r["latest_date"],
            "price": r["price"],
            "current_value_eur": r["current_value_eur"],
            "weight": r["weight"],
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        for r in rows
    ])

    snapshot.to_csv(DATA_DIR / "last_snapshot.csv", index=False)
    dated_path = SNAPSHOT_DIR / f"snapshot_{date.today().isoformat()}.csv"
    snapshot.to_csv(dated_path, index=False)


def compute_equity_curve(value_matrix: pd.DataFrame) -> list[dict]:
    daily_total = value_matrix.sum(axis=1).dropna()
    if daily_total.empty:
        return []

    indexed = daily_total / daily_total.iloc[0] * 100
    indexed = indexed.dropna()

    # Keep all business-day points, front-end handles filtering ALL/1Y/6M/1M.
    return [
        {"date": idx.date().isoformat(), "value": round(float(val), 4)}
        for idx, val in indexed.items()
    ]


def pct_return(current, base):
    if current is None or base is None or base == 0:
        return None
    return (current / base - 1.0) * 100


def main():
    settings = json.loads((DATA_DIR / "portfolio_settings.json").read_text(encoding="utf-8"))
    start_date = settings["start_date"]

    positions = pd.read_csv(DATA_DIR / "positions.csv")
    master = pd.read_csv(DATA_DIR / "instruments_master.csv")
    df = positions.merge(master, on="instrument_id", how="left", validate="many_to_one")

    previous_snapshot = load_previous_snapshot()
    previous_by_id = {}
    if previous_snapshot is not None:
        previous_by_id = {
            row["instrument_id"]: row.to_dict()
            for _, row in previous_snapshot.iterrows()
        }

    rows = []
    errors = []
    value_series_map = {}

    now = pd.Timestamp.today()
    month_start = pd.Timestamp(date(now.year, now.month, 1))
    year_start = pd.Timestamp(date(now.year, 1, 1))

    for _, r in df.iterrows():
        item = r.to_dict()
        instrument_id = item["instrument_id"]
        ticker = item["ticker_yahoo"]
        quantity = float(item["quantity"])
        currency = item["currency"]

        try:
            close = download_close(ticker, start=start_date)
            fx = fx_to_eur_series(currency, start=start_date, index=close.index)
            close_eur = close * fx
            value_eur = close_eur * quantity
            value_series_map[instrument_id] = value_eur.rename(instrument_id)

            latest_price = safe_float(close.iloc[-1])
            latest_date = close.index[-1].date().isoformat()
            current_value = safe_float(value_eur.iloc[-1])

            prev_snapshot = previous_by_id.get(instrument_id)
            previous_value = safe_float(prev_snapshot.get("current_value_eur")) if prev_snapshot else None
            last_update_return = pct_return(current_value, previous_value)

            month_base = first_after_or_equal(value_eur, month_start)
            ytd_base = first_after_or_equal(value_eur, year_start)

            month_return = pct_return(current_value, month_base)
            ytd_return = pct_return(current_value, ytd_base)
            purchase_return = pct_return(current_value, float(item["purchase_amount_eur"]))

            rows.append({
                "instrument_id": instrument_id,
                "name": item["name"],
                "isin": item["isin"],
                "ticker": ticker,
                "asset_class": item["asset_class"],
                "sector": item["sector"],
                "region": item["region"],
                "currency": currency,
                "mode": item["mode"],
                "frequency": item.get("frequency") if not pd.isna(item.get("frequency")) else "",
                "purchase_date": item["purchase_date"],
                "quantity": quantity,
                "purchase_amount_eur": float(item["purchase_amount_eur"]),
                "price": latest_price,
                "latest_date": latest_date,
                "fx_to_eur": safe_float(fx.iloc[-1]),
                "current_value_eur": current_value,
                "previous_value_eur": previous_value,
                "last_update_return": last_update_return,
                "month_return": month_return,
                "ytd_return": ytd_return,
                "purchase_return": purchase_return,
                "data_status": "OK",
            })

        except Exception as exc:
            errors.append({"instrument_id": instrument_id, "ticker": ticker, "error": str(exc)})
            rows.append({
                "instrument_id": instrument_id,
                "name": item.get("name"),
                "isin": item.get("isin"),
                "ticker": ticker,
                "asset_class": item.get("asset_class"),
                "sector": item.get("sector"),
                "region": item.get("region"),
                "currency": currency,
                "mode": item.get("mode"),
                "frequency": item.get("frequency") if not pd.isna(item.get("frequency")) else "",
                "purchase_date": item.get("purchase_date"),
                "quantity": quantity,
                "purchase_amount_eur": float(item["purchase_amount_eur"]),
                "price": None,
                "latest_date": None,
                "fx_to_eur": None,
                "current_value_eur": None,
                "previous_value_eur": None,
                "last_update_return": None,
                "month_return": None,
                "ytd_return": None,
                "purchase_return": None,
                "data_status": "ERROR",
            })

    total_value = sum(r["current_value_eur"] or 0 for r in rows)
    previous_total_value = sum(r["previous_value_eur"] or 0 for r in rows)

    for row in rows:
        row["weight"] = (row["current_value_eur"] / total_value * 100) if total_value > 0 and row["current_value_eur"] is not None else None
        row["previous_weight"] = (row["previous_value_eur"] / previous_total_value * 100) if previous_total_value > 0 and row["previous_value_eur"] is not None else None

        # Contribution from last update should ideally use previous weight.
        # If no previous snapshot exists, fallback to current weight.
        weight_for_last_update = row["previous_weight"] if row["previous_weight"] is not None else row["weight"]
        row["last_update_contribution"] = (
            (weight_for_last_update / 100) * row["last_update_return"]
            if weight_for_last_update is not None and row["last_update_return"] is not None
            else None
        )

        # Purchase contribution uses current weight as a simple dashboard proxy.
        row["purchase_contribution"] = (
            (row["weight"] / 100) * row["purchase_return"]
            if row["weight"] is not None and row["purchase_return"] is not None
            else None
        )

    portfolio_last_update_return = (
        (total_value / previous_total_value - 1.0) * 100
        if previous_total_value > 0 else None
    )

    value_matrix = pd.concat(value_series_map.values(), axis=1).dropna(how="all") if value_series_map else pd.DataFrame()
    equity_curve = compute_equity_curve(value_matrix)

    counts = {
        "total_instruments": len([r for r in rows if r["mode"] in ("PIC", "PAC")]),
        "equity_etfs": len([r for r in rows if r["asset_class"] == "ETF Equity"]),
        "single_stocks": len([r for r in rows if r["asset_class"] == "Equity"]),
        "bond_etfs": len([r for r in rows if r["asset_class"] == "ETF Bond"]),
        "commodity_etfs": len([r for r in rows if r["asset_class"] == "ETF Commodity"]),
        "crypto": len([r for r in rows if r["asset_class"] == "Crypto"]),
    }

    latest_dates = sorted({r["latest_date"] for r in rows if r.get("latest_date")})

    data = {
        "settings": settings,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "last_price_date": latest_dates[-1] if latest_dates else None,
        "portfolio": {
            "name": settings["portfolio_name"],
            "base_currency": settings["base_currency"],
            "total_value_eur": total_value,
            "previous_total_value_eur": previous_total_value,
            "last_update_return": portfolio_last_update_return,
            "counts": counts,
            "holdings": rows,
            "pac": [r for r in rows if r["mode"] == "PAC"],
            "pic": [r for r in rows if r["mode"] == "PIC"],
            "asset_allocation": aggregate_weights(rows, "asset_class", total_value),
            "sector_allocation": aggregate_weights(rows, "sector", total_value),
            "currency_exposure": aggregate_weights(rows, "currency", total_value),
            "equity_curve": equity_curve,
            "errors": errors,
            "data_status": {
                "ok": len(errors) == 0,
                "missing_or_error": len(errors),
                "source": settings["data_source"],
            },
        }
    }

    (DOCS_DIR / "dashboard_data.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"])
    )
    template = env.get_template("index.html")
    html = template.render(dashboard_data=json.dumps(data, ensure_ascii=False))

    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")

    save_snapshot(rows)

    print(f"Generated dashboard at docs/index.html")
    print(f"Total value EUR: {total_value:,.2f}")
    print(f"Last price date: {data['last_price_date']}")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
