import pandas as pd
import yfinance as yf
import pickle
import datetime
from datetime import datetime, timedelta
import random # 임의 날짜 선택을 위해 추가
import numpy as np

from river import preprocessing
from river import forest
from river import compose
from river import metrics

# 참조에 있는 모듈들을 직접 import하여 사용합니다. (더미 정의 없음)
from rise_proto import initialize_dart_data, get_full_financial_report
from log1p import Log1pTransformer
from constants import (
    log_features_common, log_features_with_financials,
    feature_columns_no_financials, feature_columns_with_financials,
    MODEL_NO_FINANCIALS_SAVE_PATH, MODEL_WITH_FINANCIALS_SAVE_PATH,
    base_feature_columns, lagged_base_features, financial_feature_columns_list,
    DART_API_KEY, KOREAN_STOCK_TICKERS, INTERNATIONAL_STOCK_TICKERS,
    FULL_DATA_FETCH_RANGE, STOCK_PERIOD, LAG_PERIODS
)
from get_data import get_processed_features_for_stock


# --- Global Settings ---
PERIOD = '1d'
LAG_PERIODS_INT = 90 # '90d'를 정수 90으로 변환
STOCK_TICKERS = ["AAPL", "MSFT", "GOOGL", "005930.KS", "000660.KS", "035420.KS", "000020.KS", "005380.KS"]

# 날짜 범위 정의
START_DATE_RANGE = datetime(2020, 1, 1).replace(hour=0, minute=0, second=0, microsecond=0)
END_DATE_RANGE = datetime(2025, 10, 24).replace(hour=0, minute=0, second=0, microsecond=0) # 2025년 10월 24일 포함

# 각 종목별 반복 횟수
NUM_ITERATIONS_PER_STOCK = 500

# --- Helper function to get a random date within the range ---
def get_random_date_in_range(start_date: datetime, end_date: datetime) -> datetime:
    """주어진 날짜 범위 내에서 임의의 날짜를 반환합니다."""
    time_delta = end_date - start_date
    random_days = random.randrange(time_delta.days + 1)
    return start_date + timedelta(days=random_days)


# --- DART API Initialization ---
dart_initialized_successfully = False
try:
    initialize_dart_data(DART_API_KEY)
    dart_initialized_successfully = True
    print("DART API 및 기업 코드 초기화 성공.")
except Exception as e:
    print(f"DART API 및 기업 코드 초기화 실패: {e}")
    dart_initialized_successfully = False


# --- River OML Model Configuration ---

# 학습에 사용될 모든 특징 컬럼 정의
all_input_features = list(set(
    base_feature_columns +
    [f'Close_lag_{i}' for i in range(1, LAG_PERIODS_INT + 1)] +
    financial_feature_columns_list
))

# OML 모델 파이프라인 생성 (ARFRegressor의 n_models 수정됨)
model_pipeline = compose.Pipeline(
    ('scaler', preprocessing.StandardScaler()),
    ('regressor', forest.ARFRegressor(seed=42, n_models=10))
)

# OML 학습을 위한 지표 (모든 종목의 모든 시행에 걸쳐 누적될 단일 지표)
metric = metrics.MAE()


# --- River OML Model Training ---
print("\n--- River OML 단일 모델, 종목별 순차 학습 시작 (임의 날짜 반복) ---")
print(f"날짜 범위: {START_DATE_RANGE.strftime('%Y-%m-%d')} ~ {END_DATE_RANGE.strftime('%Y-%m-%d')}")
print(f"각 종목당 {NUM_ITERATIONS_PER_STOCK}회 반복 시행.")

total_processed_data_points = 0
skipped_data_points = 0 # 건너뛴 데이터 포인트 수

# 모든 종목의 full_hist_raw를 미리 가져와서 캐싱 (매번 Yahoo Finance 호출 방지)
cached_full_hists = {}
for ticker_symbol in STOCK_TICKERS:
    print(f"[{ticker_symbol}] 이력 데이터 로딩 중...")
    full_hist_raw = yf.Ticker(ticker_symbol).history(period='max', interval=PERIOD)
    if full_hist_raw.empty:
        print(f"[{ticker_symbol}] 이력 데이터가 없어 캐싱에서 제외됩니다.")
        continue
    full_hist_raw.index = full_hist_raw.index.tz_localize(None).normalize()
    full_hist_raw = full_hist_raw.sort_index(ascending=True)
    cached_full_hists[ticker_symbol] = full_hist_raw

for ticker_symbol in STOCK_TICKERS:
    print(f"\n--- 종목: {ticker_symbol} 학습 시작 ---")
    is_korean_stock = ticker_symbol.endswith(".KS")
    
    if ticker_symbol not in cached_full_hists:
        print(f"[{ticker_symbol}] 캐시된 이력 데이터가 없습니다. 건너뜝니다.")
        continue
    
    full_hist_raw = cached_full_hists[ticker_symbol]

    for iteration in range(NUM_ITERATIONS_PER_STOCK):
        # 매 반복마다 임의의 날짜를 새로 선택
        current_fixed_end_date = get_random_date_in_range(START_DATE_RANGE, END_DATE_RANGE)
        current_prediction_target_date = current_fixed_end_date + timedelta(days=1)
        
        # 주말/휴일 등으로 인해 다음 날 데이터가 없을 경우를 대비하여,
        # 실제로 데이터가 존재하는 다음 거래일을 찾도록 로직을 강화할 수도 있으나,
        # 여기서는 단순히 +1d로 진행합니다. (yfinance가 자동으로 주말 건너뜀)

        # 현재 날짜 선택 정보를 출력
        print(f"  [{ticker_symbol}] 반복 {iteration + 1}/{NUM_ITERATIONS_PER_STOCK}: 특징 기준 날짜 = {current_fixed_end_date.strftime('%Y-%m-%d')}, 예측 목표 날짜 = {current_prediction_target_date.strftime('%Y-%m-%d')}")

        # PREDICTION_TARGET_DATE의 'Close' 값을 가져오기 위한 유효성 검사
        # 현재 선택된 날짜가 full_hist_raw의 가장 오래된 날짜보다 더 과거일 경우도 skip
        if current_fixed_end_date < full_hist_raw.index.min():
            print(f"    [{ticker_symbol}] 특징 기준 날짜 {current_fixed_end_date.strftime('%Y-%m-%d')}가 이력 데이터({full_hist_raw.index.min().strftime('%Y-%m-%d')})보다 이전입니다. 건너뜁니다.")
            skipped_data_points += 1
            continue
        
        # current_prediction_target_date가 full_hist_raw 인덱스에 없는 경우, 가장 가까운 다음 거래일을 찾습니다.
        try:
            # get_loc(method='nearest') 대신, 직접 날짜 인덱싱을 시도하고 KeyError 발생 시 가장 가까운 유효한 날짜를 찾도록 변경
            try:
                # 먼저 정확히 일치하는 날짜를 시도
                y_true = full_hist_raw.loc[current_prediction_target_date, 'Close']
                actual_y_date = current_prediction_target_date # 실제 y_true가 사용될 날짜
            except KeyError:
                # 정확히 일치하는 날짜가 없으면, 인덱스를 사용하여 가장 가까운 다음 날짜를 찾음
                # (주말/공휴일 등으로 인해 다음 날 데이터가 없는 경우를 처리)
                # 'searchsorted'는 인덱스에 'current_prediction_target_date'가 들어갈 위치를 반환
                idx_pos = full_hist_raw.index.searchsorted(current_prediction_target_date)
                
                # searchsorted 결과가 인덱스 길이를 초과하거나, 해당 위치의 날짜가 너무 멀면 문제
                if idx_pos >= len(full_hist_raw.index):
                    # 인덱스 끝을 넘어선 경우, 데이터 없음으로 간주
                    print(f"    [{ticker_symbol}] {current_prediction_target_date.strftime('%Y-%m-%d')} 날짜 이후의 'Close' 데이터가 이력에 없습니다. 건너뜝니다.")
                    skipped_data_points += 1
                    continue
                
                # 찾은 인덱스 위치의 날짜를 실제 y_true 날짜로 사용
                actual_y_date = full_hist_raw.index[idx_pos]
                
                # 찾은 날짜가 예측 목표 날짜보다 너무 앞설 경우 (예: random_date + 1일이 주말이고, 다음 거래일이 한참 뒤인 경우)
                # 여기서는 단순히 다음 유효한 거래일이므로, 실제 목표 날짜와 같거나 그 이후여야 함
                if actual_y_date < current_prediction_target_date:
                    print(f"    [{ticker_symbol}] {current_prediction_target_date.strftime('%Y-%m-%d')} 날짜의 'Close' 데이터를 찾았으나 너무 오래된 날짜입니다 ({actual_y_date.strftime('%Y-%m-%d')}). 건너뜝니다.")
                    skipped_data_points += 1
                    continue

                y_true = full_hist_raw.loc[actual_y_date, 'Close']
        except KeyError:
            print(f"    [{ticker_symbol}] {current_prediction_target_date.strftime('%Y-%m-%d')} 날짜의 'Close' 데이터가 이력에 없습니다. 건너뜁니다.")
            skipped_data_points += 1
            continue

        # FIXED_END_DATE_FOR_FEATURES까지의 데이터를 포함하는 데이터프레임
        hist_for_features = full_hist_raw[full_hist_raw.index <= current_fixed_end_date]
        
        # 학습에 필요한 최소 데이터 포인트 확인
        if len(hist_for_features) < LAG_PERIODS_INT + 1:
            print(f"    [{ticker_symbol}] {current_fixed_end_date.strftime('%Y-%m-%d')}까지 특징 생성에 필요한 최소 데이터 포인트({LAG_PERIODS_INT + 1}일)가 부족합니다. 현재: {len(hist_for_features)}일. 건너뜝니다.")
            skipped_data_points += 1
            continue
        
        # get_processed_features_for_stock 함수는 end_date 파라미터를 str 타입으로 받습니다.
        features_df_row, has_financial = get_processed_features_for_stock(
            stock_ticker=ticker_symbol,
            end_date=current_fixed_end_date.strftime('%Y-%m-%d'), # datetime 객체를 문자열로 변환하여 전달
            period=PERIOD,
            lag_periods=LAG_PERIODS_INT,
            is_korean_stock=is_korean_stock,
            dart_initialized_successfully=dart_initialized_successfully
        )

        if features_df_row.empty or features_df_row.isnull().values.any():
            print(f"    [{ticker_symbol}] {current_fixed_end_date.strftime('%Y-%m-%d')} 날짜의 특징 데이터 생성 또는 결측치 문제로 건너뜝니다.")
            skipped_data_points += 1
            continue
        
        # 특징 데이터 (X) 준비 (River 모델 입력 형식)
        X = features_df_row.iloc[0][all_input_features].fillna(0).to_dict() # NaN 값 0으로 채우고 dict로 변환
        
        # 예측 (학습 전)
        y_pred = model_pipeline.predict_one(X)
        
        # 학습
        model_pipeline.learn_one(X, y_true)
        
        # 평가
        if y_pred is not None:
            metric.update(y_true, y_pred)
        
        total_processed_data_points += 1
        
        if total_processed_data_points % 50 == 0: # 50 데이터 포인트마다 진행 상황 출력
            print(f"    [PROGRESS] 총 처리된 데이터 포인트: {total_processed_data_points}, 현재 누적 MAE = {metric.get():.4f}")

    print(f"--- 종목: {ticker_symbol} 학습 완료. 현재 누적 MAE: {metric.get():.4f} ---")

print(f"\n--- 모든 종목에 대한 단일 모델 순차 학습 완료 ---")
print(f"총 유효 학습 데이터 포인트: {total_processed_data_points}")
print(f"총 건너뛴 데이터 포인트: {skipped_data_points}")
print(f"최종 누적 MAE: {metric.get():.4f}")

# 단일 모델 저장
model_save_path = f"initializers/river_oml_single_model_random_date_range_{START_DATE_RANGE.strftime('%Y%m%d')}_to_{END_DATE_RANGE.strftime('%Y%m%d')}.pkl"
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
    print(f"단일 모델 저장 완료: {model_save_path}")
except Exception as e:
    print(f"단일 모델 저장 실패: {e}")