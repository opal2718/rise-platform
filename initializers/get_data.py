import yfinance as yf
import pandas as pd
from rise_proto import initialize_dart_data, get_full_financial_report
from constants import MODEL_NO_FINANCIALS_SAVE_PATH, MODEL_WITH_FINANCIALS_SAVE_PATH, base_feature_columns, lagged_base_features, financial_feature_columns_list, DART_API_KEY, KOREAN_STOCK_TICKERS, INTERNATIONAL_STOCK_TICKERS, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS
import numpy as np
from datetime import datetime, timedelta

# --- Data Acquisition and Preparation (MODIFIED) ---
def get_processed_features_for_stock(stock_ticker: str, end_date: str, period: str, lag_periods: int, is_korean_stock: bool, dart_initialized_successfully: bool):
    ticker = yf.Ticker(stock_ticker)
    # 충분한 과거 데이터를 가져오기 위해 'period'를 'max'로 설정하거나,
    # 필요한 최대 기간(예: 이동 평균 계산을 위한 윈도우 크기)을 고려하여 기간을 설정합니다.
    # 여기서는 'max'로 유지하되, 이후 필요한 데이터만 필터링합니다.
    hist = ticker.history(period='max', interval=period)
    if hist.empty:
        print(f"No history data for {stock_ticker} in the specified range.")
        return pd.DataFrame(), False

    hist.index = hist.index.tz_localize(None)
    end_date_no_tz = pd.to_datetime(end_date)
    hist = hist[hist.index <= end_date_no_tz]
    hist = hist.sort_index(ascending=True)

    # 모든 피쳐 계산을 위한 충분한 과거 데이터 확보
    # 최소한 lag_periods + 이동평균 최대 윈도우 크기 + 추가 피쳐 계산에 필요한 기간 확보
    min_required_history = max(lag_periods + 1, 20) # 예를 들어, 20일 이동평균을 위한 20일 데이터

    if len(hist) < min_required_history:
        print(f"Not enough historical data for {stock_ticker} to create all features. Need at least {min_required_history} days, but only {len(hist)} available.")
        return pd.DataFrame(), False

    # --- Feature Engineering ---
    # 1. 이동 평균 (Moving Averages)
    hist['MA_5'] = hist['Close'].rolling(window=5).mean()
    hist['MA_10'] = hist['Close'].rolling(window=10).mean()
    hist['MA_20'] = hist['Close'].rolling(window=20).mean()

    # 2. 거래량 이동 평균 (Volume Moving Averages)
    hist['VMA_5'] = hist['Volume'].rolling(window=5).mean()
    hist['VMA_10'] = hist['Volume'].rolling(window=10).mean()

    # 3. 상대 강도 지수 (RSI - Relative Strength Index)
    # RSI 계산을 위한 함수 (Pandas-ta 라이브러리 등을 사용하면 더 간편)
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window-1, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    hist['RSI_14'] = calculate_rsi(hist['Close'], window=14)

    # 4. 주가 변동률 (Daily Returns)
    hist['Daily_Return'] = hist['Close'].pct_change()

    # 5. 변동성 (Volatility - Standard Deviation)
    hist['Volatility_5'] = hist['Daily_Return'].rolling(window=5).std()
    hist['Volatility_10'] = hist['Daily_Return'].rolling(window=10).std()

    # 6. 가격 추세 (Price Trend - 10일 간격 상승/하락세)
    # 10일 전 대비 현재 종가 비교
    hist['Price_Trend_10_Days'] = (hist['Close'] / hist['Close'].shift(10) - 1) * 100
    # 상승/하락 여부 이진 피쳐 (상승: 1, 하락: -1, 변동 없음: 0)
    hist['Price_Direction_10_Days'] = np.where(hist['Close'] > hist['Close'].shift(10), 1,
                                               np.where(hist['Close'] < hist['Close'].shift(10), -1, 0))

    # 7. 거래량 변화율 (Volume Change)
    hist['Volume_Change'] = hist['Volume'].pct_change()

    # 8. 최고가 대비 종가 비율 (Close to High Ratio)
    hist['Close_to_High_Ratio'] = hist['Close'] / hist['High']

    # 9. 최저가 대비 종가 비율 (Close to Low Ratio)
    hist['Close_to_Low_Ratio'] = hist['Close'] / hist['Low']


    # --- 최종 단일 행 DataFrame 생성 ---
    # 추가된 피쳐 컬럼들을 `all_feature_columns`에 반영
    additional_features = [
        'MA_5', 'MA_10', 'MA_20',
        'VMA_5', 'VMA_10',
        'RSI_14',
        'Daily_Return',
        'Volatility_5', 'Volatility_10',
        'Price_Trend_10_Days', 'Price_Direction_10_Days',
        'Volume_Change',
        'Close_to_High_Ratio', 'Close_to_Low_Ratio'
    ]

    all_feature_columns = base_feature_columns + [f'{col}_lag_{i}' for col in ['Close'] for i in range(1, lag_periods + 1)] + financial_feature_columns_list + additional_features

    final_data_index = hist.index[-1:] # 가장 최신 날짜 인덱스
    df_final_row = pd.DataFrame(index=final_data_index, columns=all_feature_columns)

    # --- 오늘 날짜의 기본 및 생성된 피쳐 값 할당 ---
    for col in base_feature_columns:
        if col in hist.columns:
            df_final_row.loc[final_data_index[0], col] = hist[col].iloc[-1]
    
    for col in additional_features:
        if col in hist.columns: # 새로 추가된 피쳐들이 hist에 있는지 확인
            df_final_row.loc[final_data_index[0], col] = hist[col].iloc[-1]

    # --- Lagged features 값 할당 (기존 로직 유지) ---
    for i in range(1, lag_periods + 1):
        df_final_row.loc[final_data_index[0], f'Close_lag_{i}'] = hist['Close'].iloc[-(i + 1)]
        # 필요한 경우 다른 기본 피쳐들도 lagged feature로 추가 가능
        # df_final_row.loc[final_data_index[0], f'Volume_lag_{i}'] = hist['Volume'].iloc[-(i + 1)]

    # Placeholder sentiment features (기존 로직 유지)
    df_final_row.loc[final_data_index[0], "stock_sentiment_avg"] = 0.0
    df_final_row.loc[final_data_index[0], "stock_news_total_count"] = 0.0
    df_final_row.loc[final_data_index[0], "sector_sentiment_avg"] = 0.0
    df_final_row.loc[final_data_index[0], "sector_relevance_avg"] = 0.0

    # --- Financial Data Integration (변동 없음) ---
    has_financial_data = False
    if is_korean_stock and dart_initialized_successfully:
        try:
            corp_identifier = stock_ticker.replace(".KS", "")
            financial_df = get_full_financial_report(corp_identifier)

            if financial_df is not None and not financial_df.empty:
                financial_series = financial_df['thstrm_amount']

                for col in financial_feature_columns_list:
                    if col in financial_series.index:
                        df_final_row.loc[final_data_index[0], col] = financial_series.loc[col]
                        has_financial_data = True
                    else:
                        df_final_row.loc[final_data_index[0], col] = np.nan
            else:
                print(f"No financial data found for {stock_ticker}. Financial features will be NaN.")
                for col in financial_feature_columns_list:
                    df_final_row.loc[final_data_index[0], col] = np.nan
        except Exception as e:
            print(f"Error fetching financial data for {stock_ticker}: {e}. Financial features will be NaN.")
            for col in financial_feature_columns_list:
                df_final_row.loc[final_data_index[0], col] = np.nan
    else:
        for col in financial_feature_columns_list:
            df_final_row.loc[final_data_index[0], col] = np.nan

    # 모든 NaN 값을 0으로 채우는 것은 최종 모델 학습 시 주의가 필요합니다.
    # 경우에 따라서는 NaN을 그대로 두거나, 다른 imputation 전략을 사용할 수도 있습니다.
    # 여기서는 기존 로직을 따라 0으로 채웁니다.
    df_final_row.fillna(0, inplace=True)
    return df_final_row, has_financial_data