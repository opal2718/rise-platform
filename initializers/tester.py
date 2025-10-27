# 코랩 셀에서 이 코드를 실행하기 전에 반드시 위의 initializer.py 코드를 먼저 실행하여
# model_with_financials, model_no_financials, feature_columns_with_financials,
# feature_columns_no_financials, LAG_PERIODS, STOCK_PERIOD,
# financial_feature_columns_list, dart_initialized_successfully, get_full_financial_report
# 등의 전역 변수와 함수가 생성되어 있어야 합니다.

import pandas as pd
import yfinance as yf
import numpy as np
import pickle
from rise_proto import initialize_dart_data
from get_data import get_processed_features_for_stock as get_features_for_prediction_test
from constants import log_features_common, log_features_with_financials, feature_columns_no_financials, feature_columns_with_financials, MODEL_NO_FINANCIALS_SAVE_PATH, MODEL_WITH_FINANCIALS_SAVE_PATH, base_feature_columns, lagged_base_features, financial_feature_columns_list, DART_API_KEY, KOREAN_STOCK_TICKERS, INTERNATIONAL_STOCK_TICKERS, FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS

MODEL_WITH_FINANCIALS_SAVE_PATH = "initializers/river_oml_single_model_fixed_date_20251022.pkl"
MODEL_NO_FINANCIALS_SAVE_PATH = "initializers/river_oml_single_model_fixed_date_20251022.pkl"

model_no_financials = pickle.load(open(MODEL_NO_FINANCIALS_SAVE_PATH, 'rb'))
model_with_financials = pickle.load(open(MODEL_WITH_FINANCIALS_SAVE_PATH, 'rb'))

try:
    initialize_dart_data(DART_API_KEY)
    dart_initialized_successfully = True
    print("DART API 및 기업 코드 초기화 성공.")
except Exception as e:
    print(f"DART API 및 기업 코드 초기화 실패: {e}")
    dart_initialized_successfully = False

# --- 모델 테스트 함수 ---
def test_initial_models_with_fresh_data():
    print("--- Starting Model Test with Fresh Yfinance Data ---")

    test_stocks = {
        "AAPL": {"is_korean": False},
        "005930.KS": {"is_korean": True},
        "000020.KS": {"is_korean": True} # 재무제표 없는 국내 주식 테스트용
    }

    results = {}

    for stock_ticker, info in test_stocks.items():
        is_korean = info["is_korean"]
        print(f"\nTesting prediction for stock: {stock_ticker} (is_korean: {is_korean})...")

        # 충분한 과거 데이터 (LAG_PERIODS + 예측에 필요한 추가 데이터)를 가져오도록 범위 설정
        required_days_for_test = globals()['LAG_PERIODS'] + 5
        fetch_range_for_test = f"{required_days_for_test}d"

        # 데이터 직접 가져오기 및 피처 엔지니어링 수행
        full_data_df, has_financials = get_features_for_prediction_test(
            stock_ticker,
            "2025-10-24",
            globals()['STOCK_PERIOD'],
            globals()['LAG_PERIODS'],
            is_korean,
            dart_initialized_successfully
        )
        """
        if full_data_df.empty or len(full_data_df) < 2: # 최소한 현재 Close와 다음 Close를 위한 2개 행 필요
            print(f"  Skipping {stock_ticker}: Not enough data available after feature engineering for prediction.")
            results[stock_ticker] = "Not enough data"
            continue"""
        
        # 시간대 처리
        full_data_df.index = pd.to_datetime(full_data_df.index)
        if full_data_df.index.tz is None:
            full_data_df.index = full_data_df.index.tz_localize('America/New_York', ambiguous='infer').tz_convert('UTC')
        else:
            full_data_df.index = full_data_df.index.tz_convert('UTC')
        full_data_df = full_data_df.sort_index()

        # 예측에 사용할 단일 인스턴스는 가장 최근의 완료된 데이터 (즉, 타겟을 만들 수 있는 마지막 행)
        # yfinance 데이터가 당일 종가를 마지막으로 포함하므로,
        # '내일'의 종가를 예측하려면 오늘 데이터까지의 피처를 사용해야 합니다.
        # 따라서 데이터프레임의 가장 마지막 행을 사용합니다.
        latest_valid_row = full_data_df.iloc[-1]

        # 예측 대상 날짜 계산
        prediction_target_date = latest_valid_row.name + pd.Timedelta(days=1)

        # 실제 다음 날 종가 (Yfinance에서 아직 가져올 수 없으므로 실제 값은 N/A)
        actual_next_close = np.nan # 예측 시점에는 실제 다음 날 종가를 알 수 없음

        # 모델 선택
        current_model = None
        current_feature_columns_for_predict = []
        model_name_used = ""

        if is_korean and has_financials:
            print(f"  {stock_ticker}: Financial data available. Using model_with_financials.")
            current_model = globals()['model_with_financials'] # 전역 변수 접근
            current_feature_columns_for_predict = globals()['feature_columns_with_financials'] # 전역 변수 접근
            model_name_used = "model_with_financials"
        elif is_korean and not has_financials:
            print(f"  {stock_ticker}: No financial data available. Using model_no_financials.")
            current_model = globals()['model_no_financials']
            current_feature_columns_for_predict = globals()['feature_columns_no_financials']
            model_name_used = "model_no_financials"
        elif not is_korean: # 해외 주식은 항상 model_no_financials 사용
            print(f"  {stock_ticker}: International stock. Using model_no_financials.")
            current_model = globals()['model_no_financials']
            current_feature_columns_for_predict = globals()['feature_columns_no_financials']
            model_name_used = "model_no_financials"

        if current_model is None:
            print(f"  Error: Could not determine model for {stock_ticker}.")
            results[stock_ticker] = "Model determination error"
            continue

        # 예측에 사용할 피처 딕셔너리 생성 (None -> 0.0 변환 포함)
        processed_X_for_prediction = {}
        for col in current_feature_columns_for_predict:
            value = latest_valid_row.get(col)
            if pd.isna(value) or value is None: # None 또는 NaN 값을 0.0으로 변환
                processed_X_for_prediction[col] = 0.0
            else:
                processed_X_for_prediction[col] = float(value)

        print(f"  Processed features for prediction: {processed_X_for_prediction}") # 디버깅용


        # 예측 수행
        try:
            predicted_value = current_model.predict_one(processed_X_for_prediction)

            print(f"  Model Used: {model_name_used}")
            print(f"  Prediction for {prediction_target_date.strftime('%Y-%m-%d')} Close: {predicted_value:.2f}")
            print(f"  Actual Close for {prediction_target_date.strftime('%Y-%m-%d')}: {actual_next_close:.2f} (N/A at prediction time)")

            results[stock_ticker] = {
                "model_used": model_name_used,
                "predicted_close": predicted_value,
                "prediction_target_date": prediction_target_date.strftime('%Y-%m-%d'),
                "actual_next_close": "N/A" # 예측 시점에는 실제 값을 알 수 없음
            }

        except Exception as e:
            print(f"  Error during prediction for {stock_ticker}: {e}")
            results[stock_ticker] = f"Prediction error: {e}"
            continue

    print("\n--- Model Test Complete ---")
    return results

# 함수 실행 (코랩 셀에서 `initializer.py` 실행 후)
test_results = test_initial_models_with_fresh_data()
print("\nFinal Test Results:")
for stock, result in test_results.items():
    print(f"  {stock}: {result}")