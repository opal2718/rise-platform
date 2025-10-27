import yfinance as yf
import pandas as pd 
from rise_proto import initialize_dart_data, get_full_financial_report 
from constants import MODEL_NO_FINANCIALS_SAVE_PATH, MODEL_WITH_FINANCIALS_SAVE_PATH, base_feature_columns, lagged_base_features, financial_feature_columns_list, DART_API_KEY, KOREAN_STOCK_TICKERS, INTERNATIONAL_STOCK_TICKERS, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS
import numpy as np
from datetime import datetime, timedelta

# --- Data Acquisition and Preparation (MODIFIED) ---
def get_processed_features_for_stock(stock_ticker: str, end_date: str, period: str, lag_periods: int, is_korean_stock: bool, dart_initialized_successfully: bool):
    # 필요한 최소한의 데이터만 가져옵니다: 현재 + lag_periods 만큼의 과거
    # 실제 거래일수를 고려하여 넉넉하게 기간을 설정합니다.
    # 예: lag_periods가 5라면, 최소 6거래일이 필요합니다. 주말 등을 고려하여 약 2주치 (14일) 정도를 가져오면 안전합니다.

    ticker = yf.Ticker(stock_ticker)
    hist = ticker.history(period='max', interval=period)
    hist.index = hist.index.tz_localize(None)
    end_date_no_tz = end_date#.replace(hour=0, minute=0, second=0, microsecond=0)
    hist = hist[hist.index <= end_date_no_tz]

    if hist.empty:
        print(f"No history data for {stock_ticker} in the specified range.")
        return pd.DataFrame(), False
    
    hist.index = hist.index.tz_localize(None)

    # 가장 최신 데이터를 가져옵니다.
    # hist는 시간 역순이 아닐 수 있으므로, 최신 데이터를 기준으로 정렬합니다.
    hist = hist.sort_index(ascending=True)

    # 필요한 과거 데이터가 충분한지 확인
    if len(hist) < lag_periods + 1:
        print(f"Not enough historical data for {stock_ticker} to create {lag_periods} lagged features. Only {len(hist)} days available.")
        return pd.DataFrame(), False

    # 최종적으로 반환할 단일 행 DataFrame을 생성합니다.
    # 컬럼은 base_feature_columns, lagged_base_features, financial_feature_columns_list를 사용합니다.
    # 초기화는 비어있는 상태로 합니다.
    #all_feature_columns = base_feature_columns + [f'{col}_lag_{i}' for col in ['Close', 'Volume'] for i in range(1, lag_periods + 1)] + financial_feature_columns_list
    all_feature_columns = base_feature_columns + [f'{col}_lag_{i}' for col in ['Close'] for i in range(1, lag_periods + 1)] + financial_feature_columns_list

    # 단일 행 DataFrame을 생성하고 인덱스는 가장 최신 날짜로 설정합니다.
    final_data_index = hist.index[-1:] # 가장 최신 날짜 인덱스
    df_final_row = pd.DataFrame(index=final_data_index, columns=all_feature_columns)
    print(final_data_index)
    # --- 오늘 날짜의 'Close', 'Volume' 값 할당 ---
    df_final_row.loc[final_data_index[0], 'Close'] = hist['Close'].iloc[-1]
    #df_final_row.loc[final_data_index[0], 'Volume'] = hist['Volume'].iloc[-1]

    # --- Lagged features 값 할당 (shift() 없이 개별적으로 값 가져오기) ---
    for i in range(1, lag_periods + 1):
        # i일 전의 데이터는 hist.iloc[-(i+1)] 에 있습니다.
        # 즉, 오늘이 -1, 어제가 -2, 2일 전이 -3 ... 입니다.
        df_final_row.loc[final_data_index[0], f'Close_lag_{i}'] = hist['Close'].iloc[-(i + 1)]
        #df_final_row.loc[final_data_index[0], f'Volume_lag_{i}'] = hist['Volume'].iloc[-(i + 1)]

    # Placeholder sentiment features
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
                        # 재무 데이터는 단일 행에만 적용되므로 직접 할당
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
            
    # 모든 값이 제대로 할당되었는지 마지막 확인 (NaN이 남아있을 수 있는 경우)
    if df_final_row.isnull().values.any():
        # 중요한 필수 컬럼(Close, Volume, lagged features)에 NaN이 있으면 문제
        # 여기서는 df_final_row를 초기화할 때 이미 NaN으로 채워지므로,
        # 할당되지 않은 컬럼이 여전히 NaN일 수 있습니다.
        # 하지만 핵심은 필수 데이터가 채워졌는지 여부이므로, 이 부분은 유연하게 처리합니다.
        pass
    df_final_row.fillna(0, inplace=True)
    return df_final_row, has_financial_data