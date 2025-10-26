import yfinance as yf
import pandas as pd 
from rise_proto import initialize_dart_data, get_full_financial_report 
from constants import MODEL_NO_FINANCIALS_SAVE_PATH, MODEL_WITH_FINANCIALS_SAVE_PATH, base_feature_columns, lagged_base_features, financial_feature_columns_list, DART_API_KEY, KOREAN_STOCK_TICKERS, INTERNATIONAL_STOCK_TICKERS, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS
import numpy as np


# --- Data Acquisition and Preparation (MODIFIED) ---
def get_processed_features_for_stock(stock_ticker: str, fetch_range: str, period: str, lag_periods: int, is_korean_stock: bool, dart_initialized_successfully: bool):
    ticker = yf.Ticker(stock_ticker)
    hist = ticker.history(period=fetch_range, interval=period)

    if hist.empty:
        print(f"No history data for {stock_ticker}")
        return pd.DataFrame(), False

    df_trend = hist[['Close', 'Volume']].copy()

    # Lag features
    lagged_features = []
    for i in range(1, lag_periods + 1):
        lagged_features.append(df_trend['Close'].shift(i).rename(f'Close_lag_{i}'))
        lagged_features.append(df_trend['Volume'].shift(i).rename(f'Volume_lag_{i}'))

    if lagged_features:
        df_lagged = pd.concat(lagged_features, axis=1)
        df_trend = pd.concat([df_trend, df_lagged], axis=1)

    # Placeholder sentiment features
    df_trend["stock_sentiment_avg"] = 0.0
    df_trend["stock_news_total_count"] = 0.0
    df_trend["sector_sentiment_avg"] = 0.0
    df_trend["sector_relevance_avg"] = 0.0

    # --- Financial Data Integration (MODIFIED) ---
    has_financial_data = False
    if is_korean_stock and dart_initialized_successfully:
        try:
            # 주가 티커에서 '.KS'를 제거하고 corp_identifier로 사용
            corp_identifier = stock_ticker.replace(".KS", "")
            # get_full_financial_report는 가장 최근 연도 (bsns_year=None)의 재무 정보를 가져옵니다.
            # 이 함수는 DataFrame을 반환하며, 인덱스는 'row_name', 컬럼은 'thstrm_amount'입니다.
            financial_df = get_full_financial_report(corp_identifier)

            if financial_df is not None and not financial_df.empty:
                # financial_df의 인덱스(row_name)를 컬럼으로 변환하고, 값을 가져옵니다.
                # 'thstrm_amount' 컬럼의 값을 사용합니다.
                financial_series = financial_df['thstrm_amount']

                # df_trend의 모든 행에 재무 데이터를 병합합니다.
                # 주의: 재무 데이터는 시계열 데이터가 아니므로, 모든 타임스텝에 동일하게 적용됩니다.
                # 실제 시계열 재무 데이터를 사용하려면, bsns_year를 df_trend의 날짜에 맞게 반복적으로 조회해야 합니다.
                # 현재는 단일 연도 데이터를 모든 행에 반복합니다.
                for col in financial_feature_columns_list:
                    if col in financial_series.index:
                        df_trend[col] = financial_series.loc[col]
                        has_financial_data = True
                    else:
                        df_trend[col] = np.nan # 해당 피처가 없으면 NaN으로 채움
            else:
                print(f"No financial data found for {stock_ticker}. Financial features will be NaN.")
                for col in financial_feature_columns_list:
                    df_trend[col] = np.nan
        except Exception as e:
            print(f"Error fetching financial data for {stock_ticker}: {e}. Financial features will be NaN.")
            for col in financial_feature_columns_list:
                df_trend[col] = np.nan
    else:
        # 비한국 주식 또는 DART 초기화 실패 시 재무 피처를 NaN으로 채움
        for col in financial_feature_columns_list:
            df_trend[col] = np.nan

    df_trend = df_trend.dropna(subset=['Close', 'Volume'])

    # has_financial_data 플래그를 반환하여, 훈련 루프에서 어떤 모델을 사용할지 결정할 수 있도록 합니다.
    return df_trend, has_financial_data
