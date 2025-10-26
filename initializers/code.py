import pandas as pd
import yfinance as yf
import pickle
import datetime
from river import preprocessing
from river.forest import ARFRegressor
import os
import numpy as np

from rise_proto import initialize_dart_data, get_full_financial_report 
from log1p import Log1pTransformer
from constants import MODEL_NO_FINANCIALS_SAVE_PATH, MODEL_WITH_FINANCIALS_SAVE_PATH, base_feature_columns, lagged_base_features, financial_feature_columns_list, DART_API_KEY, KOREAN_STOCK_TICKERS, INTERNATIONAL_STOCK_TICKERS, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS
from get_data import get_processed_features_for_stock

# --- Mount Google Drive ---
#drive.mount('/content/drive')
#print("Google Drive mounted.")


try:
    initialize_dart_data(DART_API_KEY)
    dart_initialized_successfully = True
    print("DART API 및 기업 코드 초기화 성공.")
except Exception as e:
    print(f"DART API 및 기업 코드 초기화 실패: {e}")
    dart_initialized_successfully = False


# --- Model Setup (MODIFIED) ---
if __name__ == "__main__":
    #os.makedirs(GD_ROOT, exist_ok=True)
    #os.makedirs(MODEL_CHECKPOINT_DIR, exist_ok=True)

    # ✅ Log transform both Volume & financials
    log_features_common = ['Volume'] + [f'Volume_lag_{i}' for i in range(1, LAG_PERIODS + 1)]
    log_features_with_financials = financial_feature_columns_list + log_features_common

    # ✅ Nonlinear model (Adaptive Random Forest)
    model_with_financials = (
        Log1pTransformer(features_to_transform=log_features_with_financials) |
        preprocessing.StandardScaler() |
        ARFRegressor(n_models=10, seed=42)
    )

    model_no_financials = (
        Log1pTransformer(features_to_transform=log_features_common) |
        preprocessing.StandardScaler() |
        ARFRegressor(n_models=10, seed=42)
    )

    # --- Feature Columns ---

    feature_columns_with_financials = base_feature_columns + lagged_base_features + financial_feature_columns_list
    feature_columns_no_financials = base_feature_columns + lagged_base_features

    print("\n--- Starting Model Training ---")

    all_tickers = KOREAN_STOCK_TICKERS + INTERNATIONAL_STOCK_TICKERS

    for stock_ticker in all_tickers:
        print(f"\nProcessing {stock_ticker}...")
        is_korean = stock_ticker.endswith(".KS")

        # get_processed_features_for_stock에서 재무 데이터 유무 플래그도 함께 반환
        df, current_stock_has_financial_data = get_processed_features_for_stock(
            stock_ticker, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS, is_korean, dart_initialized_successfully
        )

        if df.empty:
            print(f"No data for {stock_ticker}.")
            continue

        df['target_next_close'] = df['Close'].shift(-1)
        df = df.dropna(subset=['target_next_close'])

        if df.empty:
            print(f"No target data for {stock_ticker} after dropping NaNs.")
            continue

        # 어떤 모델을 훈련할지 결정
        if current_stock_has_financial_data and is_korean: # 한국 주식이고 실제 재무 데이터가 있다면 with_financials 모델 훈련
            print(f"Training 'model_with_financials' for {stock_ticker} (with financial data).")
            current_model = model_with_financials
            current_feature_columns = feature_columns_with_financials
        else: # 그 외의 경우 (비한국 주식, 한국 주식인데 재무 데이터 없음) no_financials 모델 훈련
            print(f"Training 'model_no_financials' for {stock_ticker} (no financial data or international).")
            current_model = model_no_financials
            current_feature_columns = feature_columns_no_financials

        for _, row in df.iterrows():
            # 현재 훈련할 모델에 맞는 피처만 선택
            X = row[current_feature_columns].to_dict()
            y = row['target_next_close']

            # NaN 값을 0.0으로 변환 (Log1pTransformer는 음수를 처리하지 않으므로, 0.0 이상의 값으로 유지)
            for k in X:
                if pd.isna(X[k]):
                    X[k] = 0.0

            current_model.learn_one(X, float(y))

        print(f"Done training on {stock_ticker} ({len(df)} samples).")

    # Save models
    with open(MODEL_WITH_FINANCIALS_SAVE_PATH, 'wb') as f:
        pickle.dump(model_with_financials, f)
    with open(MODEL_NO_FINANCIALS_SAVE_PATH, 'wb') as f:
        pickle.dump(model_no_financials, f)
    print("✅ Models saved successfully.")