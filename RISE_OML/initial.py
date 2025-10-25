import pandas as pd
import yfinance as yf
import pickle
import datetime
from river import linear_model
from river import preprocessing
import os
import numpy as np
from google.colab import drive

# --- Mount Google Drive ---
drive.mount('/content/drive')
print("Google Drive mounted.")

# --- Configuration ---
STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V",
    "JNJ", "PG", "UNH", "HD", "MA", "DIS", "NFLX", "ADBE", "CRM", "XOM",
    "CVX", "KO", "PEP", "WMT", "NKE", "AMD", "INTC", "CSCO", "CMCSA", "T"
]

FULL_DATA_FETCH_RANGE = "5y"
STOCK_PERIOD = "1d"

GD_ROOT = "/content/drive/MyDrive/Stock_OML_Model_Training"
MODEL_SAVE_PATH = os.path.join(GD_ROOT, "initial_universal_stock_model.pkl")

MODEL_CHECKPOINT_DIR = os.path.join(GD_ROOT, "model_checkpoints")
SAVE_INTERVAL_STOCKS = 5

TRAINING_WINDOW_TIMEDELTA = pd.Timedelta(days=90)
# 새로 추가: 지연 피처를 만들 기간 (훈련 윈도우와 일치시키는 것이 합리적)
LAG_PERIODS = 90


# --- Data Acquisition and Preparation Function (MODIFIED) ---
def get_processed_features_for_stock(stock: str, fetch_range: str, period: str, lag_periods: int):
    ticker = yf.Ticker(stock)
    hist = ticker.history(period=fetch_range, interval=period)

    if hist.empty:
        return pd.DataFrame()

    df_trend = hist[['Close', 'Volume']].copy() # Start with a fresh copy

    # --- FIX START ---
    # Create a list to hold all lagged feature Series
    lagged_features = []

    for i in range(1, lag_periods + 1):
        # Create lagged Close Series and rename it
        lagged_close = df_trend['Close'].shift(i)
        lagged_close.name = f'Close_lag_{i}' # Assign name here

        # Create lagged Volume Series and rename it
        lagged_volume = df_trend['Volume'].shift(i)
        lagged_volume.name = f'Volume_lag_{i}' # Assign name here

        lagged_features.append(lagged_close)
        lagged_features.append(lagged_volume)

    # Concatenate all lagged features into a single DataFrame in one go
    if lagged_features: # Only concatenate if there are lagged features
        df_lagged = pd.concat(lagged_features, axis=1)
        # Join the original df_trend with the new df_lagged
        df_trend = pd.concat([df_trend, df_lagged], axis=1)
    # --- FIX END ---


    # Placeholder features (unchanged)
    df_trend["stock_sentiment_avg"] = 0.0
    df_trend["stock_news_total_count"] = 0.0
    df_trend["sector_sentiment_avg"] = 0.0
    df_trend["sector_relevance_avg"] = 0.0

    # Drop rows with NaN values resulting from shifting
    df_trend = df_trend.dropna()

    processed_features_df = df_trend.copy() # Make a final copy if you want a contiguous block for later ops

    return processed_features_df


# --- Main script to generate the model using walk-forward OML with rolling window ---
if __name__ == "__main__":
    if not os.path.exists(GD_ROOT):
        os.makedirs(GD_ROOT)
        print(f"Created Google Drive root directory: {GD_ROOT}")

    if not os.path.exists(MODEL_CHECKPOINT_DIR):
        os.makedirs(MODEL_CHECKPOINT_DIR)
        print(f"Created Google Drive checkpoint directory: {MODEL_CHECKPOINT_DIR}")

    model = preprocessing.StandardScaler()
    model |= linear_model.LinearRegression()

    # --- MODIFIED: feature_columns 정의 (Lagged Features 포함) ---
    feature_columns = ['Close', 'Volume', 'stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']
    for i in range(1, LAG_PERIODS + 1):
        feature_columns.append(f'Close_lag_{i}')
        feature_columns.append(f'Volume_lag_{i}')


    total_updates_made = 0
    stocks_processed_count = 0

    for stock_ticker in STOCK_TICKERS:
        print(f"\nProcessing data for stock: {stock_ticker}...")
        # MODIFIED: LAG_PERIODS 인자 추가
        full_data_df = get_processed_features_for_stock(stock_ticker, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS)

        if full_data_df.empty:
            print(f"Skipping {stock_ticker} due to no data after feature engineering.") # 메시지 수정
            continue

        full_data_df.index = pd.to_datetime(full_data_df.index)
        if full_data_df.index.tz is None:
            full_data_df.index = full_data_df.index.tz_localize('America/New_York', ambiguous='infer').tz_convert('UTC')
        else:
            full_data_df.index = full_data_df.index.tz_convert('UTC')

        full_data_df = full_data_df.sort_index()

        full_data_df['target_next_close'] = full_data_df['Close'].shift(-1)
        full_data_df = full_data_df.dropna(subset=['target_next_close']).copy() # 타겟도 NaN이 되면 제거

        # IMPORTANT: LAG_PERIODS만큼 데이터를 Dropna 했으므로, 여기서 min len 체크 기준도 조정이 필요할 수 있습니다.
        # 최소한 TRAINING_WINDOW_TIMEDELTA + LAG_PERIODS 만큼의 원본 데이터가 있어야 합니다.
        if len(full_data_df) < (LAG_PERIODS + 1 + 1): # 피처 엔지니어링 후 최소한 2개 이상의 유효한 행(X, y)
             print(f"Skipping {stock_ticker}: Not enough valid data after processing for training window and lagged features.")
             continue

        # 훈련 윈도우 시작점 계산 로직은 그대로 유지 (이제 각 X는 90일치 정보를 담고 있음)
        first_possible_end_time = full_data_df.index[0] + TRAINING_WINDOW_TIMEDELTA

        start_idx_for_rolling = full_data_df.index.searchsorted(first_possible_end_time, side='left')

        if start_idx_for_rolling == len(full_data_df):
            print(f"Skipping {stock_ticker}: No data found after the initial training window start point.")
            continue

        if start_idx_for_rolling >= len(full_data_df) - 1:
            print(f"Skipping {stock_ticker}: Not enough data after initial window to perform predictions.")
            continue

        for i in range(start_idx_for_rolling, len(full_data_df) - 1):
            current_data_point_time = full_data_df.index[i]
            window_start_time = current_data_point_time - TRAINING_WINDOW_TIMEDELTA

            current_training_window_df = full_data_df.loc[window_start_time : current_data_point_time]

            if current_training_window_df.empty:
                continue

            for j, row_in_window in current_training_window_df.iterrows():
                # X_instance는 이제 90일치 정보를 포함하는 단일 시점의 피처 벡터가 됩니다.
                X_instance = row_in_window[feature_columns].to_dict()
                y_target = row_in_window['target_next_close']

                processed_X = {}
                for key, value in X_instance.items():
                    if pd.isna(value):
                        processed_X[key] = 0.0
                    else:
                        processed_X[key] = float(value)

                model.learn_one(processed_X, float(y_target))
                total_updates_made += 1

        print(f"Finished processing {stock_ticker}. Total OML updates so far: {total_updates_made}.")
        stocks_processed_count += 1

        if stocks_processed_count % SAVE_INTERVAL_STOCKS == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = os.path.join(MODEL_CHECKPOINT_DIR, f"model_checkpoint_after_{stocks_processed_count}_stocks_{timestamp}.pkl")
            with open(checkpoint_filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Checkpoint saved to {checkpoint_filename} on Google Drive after processing {stocks_processed_count} stocks.")

    print(f"\nInitial universal OML model training complete. Total updates across all stocks: {total_updates_made}.")

    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model, f)

    print(f"Trained universal OML model saved to {MODEL_SAVE_PATH} on Google Drive.")

    # ... (Post-training checks 코드는 위와 동일하게 유지, 여기서는 생략) ...