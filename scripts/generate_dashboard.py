from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict

import pandas as pd
import yfinance as yf
from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
TEMPLATES_DIR = ROOT / "templates"

DOCS_DIR.mkdir(exist_ok=True)


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
    """
    Download adjusted close prices from Yahoo Finance through yfinance.
    """
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
    """
    Return conversion series: 1 unit of local currency -> EUR.
    Yahoo convention:
    EURUSD=X = USD per 1 EUR.
    Therefore USD -> EUR = 1 / EURUSD.
    """
    if currency == "EUR":
        return pd.Series(1.0, index=index)

    fx_ticker = FX_TICKERS.get(currency)
    if not fx_ticker:
        raise ValueError(f"FX ticker missing for currency {currency}")

    eur_to_ccy = download_close(fx_ticker, start=start)
    ccy_to_eur = 1.0 / eur_to_ccy

    return ccy_to_eur.reindex(index, method="ffill")


def first_after_or_equal(series: pd.Series, d: pd.Timestamp):
    """
    First available value on or after a given date.
    Useful because financial series do not have data for every calendar day.
    """
    s = series[series.index >= d]
    if s.empty:
        return None
    return safe_float(s.iloc[0])


def pct_return(current, base):
    """
    Return in percentage points.
    Example:
    current/base - 1 = 0.10 -> 10.0
    """
    if current is None or base is None or base == 0:
        return None
    return (current / base - 1.0) * 100


def aggregate_weights(rows: list[dict], key: str, total_value: float) -> dict:
    """
    Aggregate current weights by asset class, sector, currency, etc.
    Returns values in percentage points.
    """
    out: Dict[str, float] = {}

    for row in rows:
        label = row.get(key) or "Other"
        value = row.get("current_value_eur") or 0.0
        out[label] = out.get(label, 0.0) + value

    if total_value <= 0:
        return {}

    weights = {k: v / total_value * 100 for k, v in out.items()}
    return dict(sorted(weights.items(), key=lambda x: x[1], reverse=True))


def compute_equity_curve(value_matrix: pd.DataFrame) -> list[dict]:
    """
    Portfolio equity curve indexed to 100 from the start date.

    Important:
    instruments have different calendars:
    - stocks/ETF trade on business days;
    - crypto trades every day;
    - FX can have a different calendar.

    To avoid artificial weekend/holiday drops, all series are reindexed
    to a common daily calendar and forward-filled before summing.
    """
    if value_matrix.empty:
        return []

    value_matrix = value_matrix.sort_index()

    start = value_matrix.index.min()
    end = value_matrix.index.max()

    daily_index = pd.date_range(start=start, end=end, freq="D")

    aligned = value_matrix.reindex(daily_index).ffill()

    # Drop dates where no instrument has any value yet.
    aligned = aligned.dropna(how="all")

    if aligned.empty:
        return []

    daily_total = aligned.sum(axis=1, min_count=1).dropna()

    if daily_total.empty:
        return []

    indexed = daily_total / daily_total.iloc[0] * 100

    return [
        {
            "date": idx.date().isoformat(),
            "value": round(float(val), 4),
        }
        for idx, val in indexed.items()
    ]


def main():
    settings = json.loads(
        (DATA_DIR / "portfolio_settings.json").read_text(encoding="utf-8")
    )

    start_date = settings["start_date"]

    positions = pd.read_csv(DATA_DIR / "positions.csv")
    master = pd.read_csv(DATA_DIR / "instruments_master.csv")

    df = positions.merge(
        master,
        on="instrument_id",
        how="left",
        validate="many_to_one",
    )

    rows = []
    errors = []
    value_series_map = {}

    today = pd.Timestamp.today().normalize()

    one_year_ago = today - pd.DateOffset(years=1)
    six_months_ago = today - pd.DateOffset(months=6)
    one_month_ago = today - pd.DateOffset(months=1)

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

            value_12m_base = first_after_or_equal(value_eur, one_year_ago)
            value_6m_base = first_after_or_equal(value_eur, six_months_ago)
            value_1m_base = first_after_or_equal(value_eur, one_month_ago)

            return_12m = pct_return(current_value, value_12m_base)
            return_6m = pct_return(current_value, value_6m_base)
            return_1m = pct_return(current_value, value_1m_base)

            purchase_amount = float(item["purchase_amount_eur"])
            purchase_return = pct_return(current_value, purchase_amount)

            rows.append(
                {
                    "instrument_id": instrument_id,
                    "name": item["name"],
                    "isin": item["isin"],
                    "ticker": ticker,
                    "asset_class": item["asset_class"],
                    "sector": item["sector"],
                    "region": item["region"],
                    "currency": currency,
                    "mode": item["mode"],
                    "frequency": item.get("frequency")
                    if not pd.isna(item.get("frequency"))
                    else "",
                    "purchase_date": item["purchase_date"],
                    "quantity": quantity,
                    "purchase_amount_eur": purchase_amount,
                    "price": latest_price,
                    "latest_date": latest_date,
                    "fx_to_eur": safe_float(fx.iloc[-1]),
                    "current_value_eur": current_value,
                    "return_12m": return_12m,
                    "return_6m": return_6m,
                    "return_1m": return_1m,
                    "purchase_return": purchase_return,
                    "data_status": "OK",
                }
            )

        except Exception as exc:
            errors.append(
                {
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "error": str(exc),
                }
            )

            rows.append(
                {
                    "instrument_id": instrument_id,
                    "name": item.get("name"),
                    "isin": item.get("isin"),
                    "ticker": ticker,
                    "asset_class": item.get("asset_class"),
                    "sector": item.get("sector"),
                    "region": item.get("region"),
                    "currency": currency,
                    "mode": item.get("mode"),
                    "frequency": item.get("frequency")
                    if not pd.isna(item.get("frequency"))
                    else "",
                    "purchase_date": item.get("purchase_date"),
                    "quantity": quantity,
                    "purchase_amount_eur": float(item["purchase_amount_eur"]),
                    "price": None,
                    "latest_date": None,
                    "fx_to_eur": None,
                    "current_value_eur": None,
                    "return_12m": None,
                    "return_6m": None,
                    "return_1m": None,
                    "purchase_return": None,
                    "data_status": "ERROR",
                }
            )

    total_value = sum(row["current_value_eur"] or 0.0 for row in rows)

    for row in rows:
        row["weight"] = (
            row["current_value_eur"] / total_value * 100
            if total_value > 0 and row["current_value_eur"] is not None
            else None
        )

        row["contribution_6m"] = (
            (row["weight"] / 100) * row["return_6m"]
            if row["weight"] is not None and row["return_6m"] is not None
            else None
        )

        row["contribution_1m"] = (
            (row["weight"] / 100) * row["return_1m"]
            if row["weight"] is not None and row["return_1m"] is not None
            else None
        )

        row["purchase_contribution"] = (
            (row["weight"] / 100) * row["purchase_return"]
            if row["weight"] is not None and row["purchase_return"] is not None
            else None
        )

    if value_series_map:
        value_matrix = pd.concat(value_series_map.values(), axis=1).dropna(how="all")
    else:
        value_matrix = pd.DataFrame()

    equity_curve = compute_equity_curve(value_matrix)

    if not value_matrix.empty:
        daily_total = value_matrix.sum(axis=1).dropna()
    else:
        daily_total = pd.Series(dtype=float)

    if not daily_total.empty:
        current_portfolio_value = safe_float(daily_total.iloc[-1])
        base_12m = first_after_or_equal(daily_total, one_year_ago)
        portfolio_12m_return = pct_return(current_portfolio_value, base_12m)
    else:
        portfolio_12m_return = None

    counts = {
        "total_instruments": len([r for r in rows if r["mode"] in ("PIC", "PAC")]),
        "equity_etfs": len([r for r in rows if r["asset_class"] == "ETF Equity"]),
        "single_stocks": len([r for r in rows if r["asset_class"] == "Equity"]),
        "bond_etfs": len([r for r in rows if r["asset_class"] == "ETF Bond"]),
        "commodity_etfs": len(
            [r for r in rows if r["asset_class"] == "ETF Commodity"]
        ),
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
            "return_12m": portfolio_12m_return,
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
        },
    }

    (DOCS_DIR / "dashboard_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "xml"]),
    )

    template = env.get_template("index.html")
    rendered_html = template.render(
        dashboard_data=json.dumps(data, ensure_ascii=False)
    )

    (DOCS_DIR / "index.html").write_text(rendered_html, encoding="utf-8")

    print("Generated dashboard at docs/index.html")
    print(f"Total value EUR: {total_value:,.2f}")
    print(f"Portfolio 12M return: {portfolio_12m_return}")
    print(f"Last price date: {data['last_price_date']}")

    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
