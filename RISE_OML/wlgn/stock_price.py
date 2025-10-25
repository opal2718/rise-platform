"""Utility functions for retrieving stock price data with market capitalization support."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Dict, Iterable, List, Sequence

import pandas as pd

_PERIOD_ALIASES = {
    "latest": "latest",
    "day": "day",
    "1d": "day",
    "week": "week",
    "1w": "week",
    "month": "month",
    "1m": "month",
    "quarter": "quarter",
    "1q": "quarter",
    "year": "year",
    "1y": "year",
    "max": "max",
    "all": "max",
    "n": "n_days",
    "n_days": "n_days",
}


@dataclass(frozen=True)
class _FetchParams:
    period: str
    n_days: int | None
    start_date: date | None
    end_date: date
    keep_latest_only: bool


def _normalize_period(period: str, n_days: int | None) -> _FetchParams:
    normalized = _PERIOD_ALIASES.get(period.lower(), period.lower())
    today = date.today()
    keep_latest_only = False

    if normalized == "latest":
        start_date = today - timedelta(days=30)
        keep_latest_only = True
        n_days_value = None
    elif normalized == "day":
        start_date = today - timedelta(days=7)
        n_days_value = 1
    elif normalized == "week":
        start_date = today - timedelta(days=14)
        n_days_value = 7
    elif normalized == "month":
        start_date = today - timedelta(days=45)
        n_days_value = 30
    elif normalized == "quarter":
        start_date = today - timedelta(days=120)
        n_days_value = 90
    elif normalized == "year":
        start_date = today - timedelta(days=400)
        n_days_value = 365
    elif normalized == "max":
        start_date = date(1990, 1, 1)
        n_days_value = None
    elif normalized == "n_days":
        if n_days is None or n_days <= 0:
            raise ValueError("n_days must be a positive integer when period='n_days'.")
        start_date = today - timedelta(days=max(5, n_days * 2))
        n_days_value = n_days
    else:
        raise ValueError(f"Unsupported period: {period}")

    return _FetchParams(
        period=normalized,
        n_days=n_days_value,
        start_date=start_date,
        end_date=today,
        keep_latest_only=keep_latest_only,
    )


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def _trim_dataframe(df: pd.DataFrame, params: _FetchParams) -> pd.DataFrame:
    if df.empty:
        return df
    df = _ensure_datetime_index(df)
    if params.keep_latest_only or params.period == "day":
        latest_date = df.index.max().normalize()
        return df[df.index.normalize() == latest_date]
    if params.period in {"week", "month", "quarter", "year", "n_days"} and params.n_days:
        cutoff = df.index.max() - timedelta(days=params.n_days * 2)
        df = df[df.index >= cutoff]
        final_cutoff = df.index.max() - timedelta(days=params.n_days - 1)
        df = df[df.index >= final_cutoff]
    return df


def _rename_pykrx_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "종가": "Close",
        "거래량": "Volume",
        "거래대금": "Value",
        "등락률": "Change",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def _fetch_from_pykrx(code: str, params: _FetchParams) -> pd.DataFrame:
    try:
        from pykrx import stock
    except ImportError as exc:
        raise ImportError("pykrx is required for source='pykrx'. Install it via pip.") from exc

    start = params.start_date.strftime("%Y%m%d") if params.start_date else None
    end = params.end_date.strftime("%Y%m%d")

    # 기본 OHLCV 데이터
    df_price = stock.get_market_ohlcv_by_date(start, end, code)
    df_price = _rename_pykrx_columns(df_price)

    # 시가총액 데이터 추가
    df_cap = stock.get_market_cap_by_date(start, end, code)
    df_cap = df_cap.rename(columns={"시가총액": "MarketCap", "상장주식수": "SharesOutstanding"})

    df = df_price.join(df_cap[["MarketCap", "SharesOutstanding"]], how="left")
    return df


def _format_yfinance_code(code: str) -> str:
    if code.isdigit() and len(code) == 6:
        return f"{code}.KS"
    return code


def _fetch_from_yfinance(code: str, params: _FetchParams) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for source='yfinance'. Install it via pip.") from exc

    yf_code = _format_yfinance_code(code)
    start = params.start_date.strftime("%Y-%m-%d") if params.start_date else None
    end = (params.end_date + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(yf_code, start=start, end=end, progress=False, auto_adjust=False)

    if "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})

    # 발행주식수 기반 시가총액 계산
    ticker = yf.Ticker(yf_code)
    shares = ticker.info.get("sharesOutstanding")
    if shares is not None:
        df["MarketCap"] = df["Close"] * shares
        df["SharesOutstanding"] = shares

    return df


def _fetch_from_fdr(code: str, params: _FetchParams) -> pd.DataFrame:
    try:
        import FinanceDataReader as fdr
    except ImportError as exc:
        raise ImportError("FinanceDataReader is required for source='FinanceDataReader'. Install it via pip.") from exc

    start = params.start_date.strftime("%Y-%m-%d") if params.start_date else None
    end = params.end_date.strftime("%Y-%m-%d")
    df = fdr.DataReader(code, start, end)

    # FDR에는 시가총액 정보가 없으므로 None으로 채움
    df["MarketCap"] = None
    df["SharesOutstanding"] = None
    return df


_FETCHERS: Dict[str, Callable[[str, _FetchParams], pd.DataFrame]] = {
    "pykrx": _fetch_from_pykrx,
    "yfinance": _fetch_from_yfinance,
    "financedatareader": _fetch_from_fdr,
}


def _normalize_codes(codes: str | Sequence[str]) -> List[str]:
    if isinstance(codes, str):
        return [codes]
    if not isinstance(codes, Iterable):
        raise TypeError("codes must be a string or an iterable of strings.")
    normalized = []
    for code in codes:
        if not isinstance(code, str):
            raise TypeError("All stock codes must be strings.")
        normalized.append(code.strip())
    if not normalized:
        raise ValueError("At least one stock code must be provided.")
    return normalized


def get_stock_price(
    codes: str | Sequence[str],
    period: str = "day",
    source: str = "pykrx",
    *,
    n_days: int | None = None,
) -> pd.DataFrame:
    """Fetch stock price and market cap data."""
    normalized_codes = _normalize_codes(codes)
    params = _normalize_period(period, n_days)
    source_key = source.lower()
    if source_key not in _FETCHERS:
        raise ValueError(f"Unsupported source: {source}. Choose from {tuple(_FETCHERS.keys())}.")

    fetcher = _FETCHERS[source_key]
    frames: List[pd.DataFrame] = []
    collected_codes: List[str] = []

    for code in normalized_codes:
        df = fetcher(code, params)
        df = _ensure_datetime_index(df)
        df = _trim_dataframe(df, params)
        if df.empty:
            continue
        frames.append(df)
        collected_codes.append(code)

    if not frames:
        raise ValueError("No data returned for the requested parameters.")

    if len(frames) == 1:
        return frames[0]

    combined = pd.concat(frames, keys=collected_codes, names=["code", "date"])
    return combined


__all__ = ["get_stock_price"]
