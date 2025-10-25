import os
import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import pickle
from google.cloud import storage
import logging
import numpy as np
import random
import requests
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://user:password@localhost/stock_predictions")
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    stock_ticker = Column(String(10), nullable=False)
    prediction_made_at = Column(DateTime(timezone=True))
    predicted_value_for_time = Column(DateTime(timezone=True), nullable=False)
    predicted_value = Column(Float, nullable=False)
    features_json = Column(JSONB, nullable=False)
    actual_value = Column(Float)
    is_correct = Column(Boolean)
    error_margin = Column(Float)
    model_version = Column(String(50))

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    version_tag = Column(String(50), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)
    trained_at = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc))
    gcs_path = Column(String(255), nullable=False)
    performance_metric = Column(Float)
    is_active = Column(Boolean, default=False)
    update_count_since_last_outdated_save = Column(Integer, default=0)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# GCS configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-gcs-bucket-name")
GCS_MODEL_FOLDER = "" 
GCS_OUTDATED_FOLDER = "outdated" 

MODEL_WITH_FINANCIALS_FILENAME = "model_with_financials.pkl"
MODEL_NO_FINANCIALS_FILENAME = "model_no_financials.pkl"
MODEL_WITH_FINANCIALS_GCS_PATH = os.path.join(GCS_MODEL_FOLDER, MODEL_WITH_FINANCIALS_FILENAME)
MODEL_NO_FINANCIALS_GCS_PATH = os.path.join(GCS_MODEL_FOLDER, MODEL_NO_FINANCIALS_FILENAME)

# Stock list files in GCS
NASDAQ_LISTED_FILE = "nasdaqlisted.txt"
KOSPI_KOSDAQ_FILE = "data_5208_20251025.csv"

LAG_PERIODS = 90
full_feature_columns = ['Close', 'Volume', 'stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']
for i in range(1, LAG_PERIODS + 1):
    full_feature_columns.append(f'Close_lag_{i}')
    full_feature_columns.append(f'Volume_lag_{i}')

financial_feature_columns_list = [
    '유동자산CFS', '비유동자산CFS', '자산총계CFS', '유동부채CFS', '비유동부채CFS',
    '부채총계CFS', '자본금CFS', '이익잉여금CFS', '자본총계CFS', '매출액CFS',
    '영업이익CFS', '법인세차감전 순이익CFS', '당기순이익(손실)CFS', '총포괄손익CFS',
    '유동자산OFS', '비유동자산OFS', '자산총계OFS', '유동부채OFS', '비유동부채OFS',
    '부채총계OFS', '자본금OFS', '이익잉여금OFS', '자본총계OFS', '매출액OFS',
    '영업이익OFS', '법인세차감전 순이익OFS', '당기순이익(손실)OFS', '총포괄손익OFS',
    'PBR', 'PER', 'ROR'
]
full_feature_columns.extend(financial_feature_columns_list)
no_financial_feature_columns = [col for col in full_feature_columns if col not in financial_feature_columns_list]

_LOADED_MODELS = {}
_STOCK_POOL = [] # Global list to store stock tickers

OUTDATED_SAVE_THRESHOLD = 60

class Log1pTransformer:
    _supervised = False

    def __init__(self, features_to_transform=None):
        self.features_to_transform = features_to_transform

    def learn_one(self, x, y=None):
        return self

    def transform_one(self, x):
        transformed_x = x.copy()
        for feature in self.features_to_transform or []:
            if feature in transformed_x and isinstance(transformed_x[feature], (int, float)) and transformed_x[feature] >= 0:
                transformed_x[feature] = np.log1p(transformed_x[feature])
        return transformed_x

def get_gcs_blob(filepath):
    client = storage.Client()
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    return bucket.blob(filepath)

def load_model_from_gcs(filepath):
    if filepath in _LOADED_MODELS:
        logger.debug(f"Model already in cache: {filepath}")
        return _LOADED_MODELS[filepath]

    blob = get_gcs_blob(filepath)
    if not blob.exists():
        logger.error(f"Model file not found in GCS at: {filepath}")
        raise FileNotFoundError(f"Model file not found in GCS at: {filepath}")
    
    with blob.open("rb") as f:
        model = pickle.load(f)
    _LOADED_MODELS[filepath] = model
    logger.info(f"Model loaded successfully from GCS: {filepath}")
    return model

def save_model_to_gcs(model, filepath):
    blob = get_gcs_blob(filepath)
    with blob.open("wb") as f:
        pickle.dump(model, f)
    _LOADED_MODELS[filepath] = model
    logger.info(f"Model saved successfully to GCS: {filepath}")

def save_outdated_model_to_gcs(model, model_type_filename_prefix):
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')
    outdated_filename = f"{model_type_filename_prefix}_{timestamp}.pkl"
    outdated_filepath = os.path.join(GCS_OUTDATED_FOLDER, outdated_filename)
    
    blob = get_gcs_blob(outdated_filepath)
    with blob.open("wb") as f:
        pickle.dump(model, f)
    logger.info(f"Outdated model saved to GCS: {outdated_filepath}")

def determine_financial_data_presence(features_dict: dict) -> bool:
    for financial_col in financial_feature_columns_list:
        if features_dict.get(financial_col) is None:
            return False
    return True

def load_stock_pool():
    global _STOCK_POOL
    logger.info("Loading stock tickers from GCS...")
    
    # Load NASDAQ stocks
    try:
        nasdaq_blob = get_gcs_blob(NASDAQ_LISTED_FILE)
        with nasdaq_blob.open("r") as f:
            nasdaq_df = pd.read_csv(f, sep='|')
            nasdaq_tickers = nasdaq_df['Symbol'].tolist()
            _STOCK_POOL.extend(nasdaq_tickers)
        logger.info(f"Loaded {len(nasdaq_tickers)} NASDAQ tickers.")
    except Exception as e:
        logger.error(f"Error loading NASDAQ stock list from GCS: {e}")

    # Load KOSPI/KOSDAQ stocks
    try:
        kospi_kosdaq_blob = get_gcs_blob(KOSPI_KOSDAQ_FILE)
        with kospi_kosdaq_blob.open("r") as f:
            kospi_kosdaq_df = pd.read_csv(f, encoding='euc-kr')
            # Yfinance에서 한국 주식을 조회하기 위해 .KS 접미사 추가
            kospi_kosdaq_tickers = [f"{code}.KS" for code in kospi_kosdaq_df['단축코드'].astype(str).tolist()]
            _STOCK_POOL.extend(kospi_kosdaq_tickers)
        logger.info(f"Loaded {len(kospi_kosdaq_tickers)} KOSPI/KOSDAQ tickers.")
    except Exception as e:
        logger.error(f"Error loading KOSPI/KOSDAQ stock list from GCS: {e}")

    _STOCK_POOL = list(set(_STOCK_POOL)) # Remove duplicates
    logger.info(f"Total {len(_STOCK_POOL)} unique stock tickers loaded into pool.")

def fetch_data_from_ai2_api(stock_ticker: str, period: str, target_date: datetime.datetime):
    """
    ai2:8001/data 엔드포인트에서 필요한 데이터를 가져옵니다.
    target_date는 YYYY-MM-DD 형식으로, ai2 API가 특정 시점의 데이터를 제공하도록
    period와 함께 사용될 수 있지만, 현재 ai2 API는 target_date 파라미터를 직접
    받지 않고 period만 사용합니다. 따라서 여기서는 period를 '1d'로 고정하고,
    Yfinance에서 target_date에 가까운 데이터를 필터링하는 방식으로 사용합니다.
    (ai2 API는 최신 데이터만 반환하는 것으로 가정합니다)
    """
    data_url = f"http://34.16.110.5:8001/data?stock={stock_ticker}&period=1d" # '1d' 또는 '1h' 등 적절한 period
    
    try:
        response = requests.get(data_url, timeout=30) # 타임아웃 30초 설정
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        data = response.json()
        
        features = data.get("processed_features_for_prediction")
        if features:
            # ai2 API에서 받은 features에 None이 있다면 그대로 유지
            # Sentiment 관련 필드는 0으로 채워질 것이며, 재무 데이터는 없으면 None일 것임
            return features
        else:
            logger.warning(f"No 'processed_features_for_prediction' found in AI2 API response for {stock_ticker}.")
            return None
    except requests.exceptions.Timeout:
        logger.error(f"AI2 API request timed out for {stock_ticker}.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from AI2 API for {stock_ticker}: {e}")
        return None

def get_random_learning_data(stock_ticker: str, today_utc: datetime.datetime):
    """
    특정 종목에 대해 모델 학습에 필요한 (features, actual_value) 쌍을 무작위 시점에서 가져옵니다.
    """
    ticker = yf.Ticker(stock_ticker)
    
    # Yfinance에서 해당 종목의 상장일(가장 오래된 데이터)을 가져옵니다.
    # period='max'가 아닌 명시적인 start/end 날짜를 사용하여 에러 발생 가능성을 줄임.
    # 안전하게 아주 먼 과거부터 오늘까지의 데이터를 요청합니다.
    # Yfinance는 보통 UTC를 사용하므로, datetime 객체에 tzinfo를 명시해줍니다.
    end_date_for_yf_fetch = today_utc - datetime.timedelta(days=1) # 오늘 데이터는 아직 완전하지 않으므로 어제까지
    # Yfinance API 호출 시 start_date를 지정하여 period='max' 오류 회피
    # 예를 들어 2000년 1월 1일 또는 그 이전으로 충분히 먼 과거 날짜를 지정
    start_date_for_yf_fetch = datetime.datetime(1980, 1, 1, tzinfo=datetime.timezone.utc) # 충분히 먼 과거
    
    try:
        hist = ticker.history(start=start_date_for_yf_fetch.strftime('%Y-%m-%d'), 
                              end=end_date_for_yf_fetch.strftime('%Y-%m-%d'), 
                              interval="1d")
    except Exception as e:
        logger.error(f"Error fetching historical data for {stock_ticker} using yfinance start/end dates: {e}")
        return None, None, None

    if hist.empty:
        logger.error(f"No historical data found for {stock_ticker} in the specified range. Skipping.")
        return None, None, None
    
    if hist.index.tz is None:
        hist.index = hist.index.tz_localize('UTC')

    first_trade_date = hist.index.min()

    # 최소 LAG_PERIODS + 1일 (actual_value)의 데이터를 채울 수 있는 시작 지점
    earliest_possible_target_date = first_trade_date + datetime.timedelta(days=LAG_PERIODS + 1)
    
    # 학습에 사용될 수 있는 유효한 날짜 범위 설정
    # (LAG_PERIODS)일의 과거 데이터와 1일의 실제값(actual_value)을 포함해야 하므로,
    # 인덱스 상으로 LAG_PERIODS + 1 번째 이후의 날짜부터 선택 가능.
    # 그리고 오늘 이전의 날짜여야 함. (end_date_for_yf_fetch가 이미 하루 전까지를 의미)
    valid_target_dates = hist.index[
        (hist.index >= earliest_possible_target_date)
    ]

    if valid_target_dates.empty:
        logger.warning(f"No valid target dates for learning for {stock_ticker} after considering lag periods and available history. Skipping.")
        return None, None, None

    # 무작위로 하나의 target_date 선택
    random_target_date_idx = random.randint(0, len(valid_target_dates) - 1)
    predicted_value_for_time = valid_target_dates[random_target_date_idx]

    # actual_value는 predicted_value_for_time의 'Close' 값
    actual_value = hist.loc[predicted_value_for_time]['Close']

    # LAG_PERIODS에 필요한 과거 데이터를 추출 (predicted_value_for_time 직전까지)
    # predicted_value_for_time을 포함하여 LAG_PERIODS + 1 개의 데이터가 필요합니다.
    # 즉, predicted_value_for_time 기준으로 이전 LAG_PERIODS 일의 데이터와 predicted_value_for_time의 데이터
    
    # target_date를 포함하여 LAG_PERIODS + 1개의 데이터를 가져와야 합니다.
    # 예를 들어, LAG_PERIODS=90 이면, target_date 포함 91일치 데이터 필요
    # target_date는 예측의 'target'이므로, feature는 target_date-1d 까지의 데이터
    # 따라서, features에 사용될 데이터는 target_date보다 1일 전까지의 LAG_PERIODS 개
    # actual_value는 target_date의 'Close'
    
    # yfinance DataFrame에서 target_date를 포함하여 과거 LAG_PERIODS+1 개의 행을 가져옵니다.
    # `loc`을 사용하면 날짜 범위로 쉽게 필터링 가능
    
    # predicted_value_for_time을 기준으로 그 이전 LAG_PERIODS 일치 데이터와
    # predicted_value_for_time 날짜의 데이터 (actual_value)를 가져옴
    
    # 데이터프레임에서 target_date를 찾고, 그 인덱스로부터 뒤로 LAG_PERIODS + 1개 가져오기
    target_date_iloc = hist.index.get_loc(predicted_value_for_time)
    
    # 실제 피처에 사용될 데이터는 target_date_iloc 바로 전까지 LAG_PERIODS 개
    # actual_value는 target_date_iloc의 'Close'
    
    if target_date_iloc < LAG_PERIODS: # LAG_PERIODS 만큼 이전 데이터가 존재하지 않는 경우
        logger.warning(f"Not enough historical data ({target_date_iloc + 1} days including target) for {stock_ticker} to fulfill {LAG_PERIODS} lag periods for target date {predicted_value_for_time}. Skipping.")
        return None, None, None

    # 피처에 사용될 과거 데이터: target_date_iloc의 이전 LAG_PERIODS개
    # 예: target_date_iloc가 90 (즉 91번째 데이터)이면, iloc[0] ~ iloc[89] (90개)
    recent_hist_for_features = hist.iloc[target_date_iloc - LAG_PERIODS : target_date_iloc]

    if len(recent_hist_for_features) < LAG_PERIODS: # 다시 한번 확인
        logger.warning(f"After slicing, still not enough historical data ({len(recent_hist_for_features)} days) for {stock_ticker} to fulfill {LAG_PERIODS} lag periods for target date {predicted_value_for_time}. Skipping.")
        return None, None, None
    
    # AI2 API에서 재무 및 센티먼트 데이터 가져오기 (이 부분은 '최신' 데이터로 가정)
    # AI2 API는 특정 `target_date`에 대한 과거 재무/센티먼트 데이터를 제공하지 않을 수 있으므로,
    # 여기서는 AI2 API가 제공하는 가장 최신 재무/센티먼트 데이터를 사용하도록 합니다.
    # 이것이 사용자님의 의도와 다르다면 알려주세요.
    ai2_features = fetch_data_from_ai2_api(stock_ticker, "1d", today_utc) # period는 1d로 고정
    if ai2_features is None:
        logger.warning(f"Failed to fetch AI2 API features for {stock_ticker}. Skipping learning for this stock.")
        return None, None, None

    # features_json 구성
    features_dict = {}
    
    # Yfinance 주가/거래량 lag features
    # 'Close'와 'Volume'은 `recent_hist_for_features`의 가장 마지막 값 (즉, predicted_value_for_time 바로 전날 값)
    current_close_val = recent_hist_for_features['Close'].iloc[-1]
    current_volume_val = recent_hist_for_features['Volume'].iloc[-1]

    features_dict['Close'] = current_close_val
    features_dict['Volume'] = current_volume_val

    for i in range(1, LAG_PERIODS + 1):
        # 역순으로 채워넣기 (예: Close_lag_1은 가장 최근 과거, Close_lag_90은 가장 오래된 과거)
        # recent_hist_for_features는 이미 LAG_PERIODS 개를 가지고 있으므로 -i 로 접근
        features_dict[f'Close_lag_{i}'] = recent_hist_for_features['Close'].iloc[-i]
        features_dict[f'Volume_lag_{i}'] = recent_hist_for_features['Volume'].iloc[-i]

    # AI2 API에서 가져온 센티먼트 및 재무 데이터 병합
    for col in ['stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']:
        features_dict[col] = ai2_features.get(col, 0.0) # 없으면 0으로 채움
    
    for col in financial_feature_columns_list:
        features_dict[col] = ai2_features.get(col, None) # 없으면 None으로 채움

    # 모든 필수 feature_columns이 features_dict에 있는지 확인하고 없으면 None 또는 0으로 채움 (안전 장치)
    for col in full_feature_columns:
        if col not in features_dict:
            if col in financial_feature_columns_list:
                features_dict[col] = None # 재무 데이터는 None
            else:
                features_dict[col] = 0.0 # 그 외는 0.0
    
    logger.debug(f"Generated features for {stock_ticker} at {predicted_value_for_time}: {features_dict}")
    return features_dict, actual_value, predicted_value_for_time


def update_models_with_random_data(num_updates_per_run=1):
    db = SessionLocal()
    try:
        if not _STOCK_POOL:
            load_stock_pool()
        if not _STOCK_POOL:
            logger.error("No stock tickers available in the pool. Cannot perform model updates.")
            return

        logger.info("--- Starting Incremental Model Update with Random Data ---")

        active_models_info = db.query(ModelVersion).filter(ModelVersion.is_active == True).all()
        
        active_models = {}
        active_gcs_paths = {}
        active_model_entries = {}
        for info in active_models_info:
            try:
                # 모델은 한 번 로드되면 캐시됨
                active_models[info.model_type] = load_model_from_gcs(info.gcs_path)
                active_gcs_paths[info.model_type] = info.gcs_path
                active_model_entries[info.model_type] = info
            except FileNotFoundError:
                logger.error(f"Active model file not found in GCS for {info.model_type} at {info.gcs_path}. Skipping update for this model type.")
                continue
        
        if not active_models:
            logger.error("No active models successfully loaded. Cannot perform update.")
            return

        models_to_update = {
            m_type: pickle.loads(pickle.dumps(model_instance)) # 딥카피하여 원본 모델을 직접 수정하지 않음
            for m_type, model_instance in active_models.items()
        }

        updates_made_with_financials = 0
        updates_made_no_financials = 0
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        for _ in range(num_updates_per_run): # 지정된 횟수만큼 업데이트 시도
            stock_ticker = random.choice(_STOCK_POOL)
            logger.info(f"Attempting to get learning data for random stock: {stock_ticker}")
            
            features_dict, actual_value, predicted_value_for_time = get_random_learning_data(stock_ticker, now_utc)

            if features_dict is None or actual_value is None or predicted_value_for_time is None:
                logger.warning(f"Could not get valid learning data for {stock_ticker}. Skipping this iteration.")
                continue

            try:
                has_financial_data = determine_financial_data_presence(features_dict)
                model_to_update_type = "with_financials" if has_financial_data else "no_financials"
                
                if model_to_update_type not in models_to_update:
                    logger.warning(f"Active model '{model_to_update_type}' not loaded/available. Skipping update for this data.")
                    continue

                selected_model = models_to_update[model_to_update_type]
                
                # 예측값 생성 (predict_one)
                # features_dict에서 모델에 필요한 특성만 추출
                if has_financial_data:
                    selected_feature_cols = full_feature_columns
                else:
                    selected_feature_cols = no_financial_feature_columns
                
                X_for_predict_learn = {}
                data_is_complete = True
                for col in selected_feature_cols:
                    value = features_dict.get(col)
                    if value is None: # None이거나 키가 없는 경우
                        # 재무 데이터는 None 허용, 그 외는 0.0으로 대체
                        if col in financial_feature_columns_list:
                            X_for_predict_learn[col] = None
                        else:
                            X_for_predict_learn[col] = 0.0 # 기본값으로 채움
                            if value is None: # 진짜 None이었으면 경고
                                logger.warning(f"Feature '{col}' was None for prediction/learn for {stock_ticker}. Replaced with 0.0.")
                    else:
                        X_for_predict_learn[col] = value
                
                # River 모델은 DictInput을 가정하므로, 여기서 DataFrame 변환은 필요 없음
                # 단, 예측/학습 전에 Log1pTransformer를 거쳐야 한다면 추가 필요.
                # 현재 Log1pTransformer는 Prediction 클래스 바깥에서 독립적으로 작동하므로
                # 여기서 직접 적용해야 함. (기존 verify_predictions.py에는 이 부분이 없었음)
                # 만약 모델 파이프라인에 Log1pTransformer가 포함되어 있다면, 모델이 알아서 처리할 것임.
                # 여기서는 raw features_dict를 바로 모델에 전달하는 것으로 가정합니다.
                
                predicted_value = selected_model.predict_one(X_for_predict_learn)

                logger.debug(f"Learning from {stock_ticker} at {predicted_value_for_time} using {model_to_update_type} model. Predicted: {predicted_value:.4f}, Actual: {actual_value:.4f}")
                selected_model.learn_one(X_for_predict_learn, actual_value)

                for key, value in features_dict.items():
                    if isinstance(value, np.integer): # Covers int64, int32, etc.
                        features_dict[key] = int(value)
                    elif isinstance(value, np.floating): # Covers float64, float32, etc.
                        features_dict[key] = float(value)
                    # Add other numpy types if you encounter them (e.g., np.bool_)
                    elif isinstance(value, datetime.datetime):
                        features_dict[key] = value.isoformat() # or str(value) if just for logging

                # Prediction 테이블에 학습 이력을 기록 (선택 사항, 요청에는 없었지만 유용할 수 있음)
                # 예측 검증 부분은 삭제되었으므로, 여기서 Prediction 객체를 생성하여 DB에 저장하는 것은
                # 모델 학습 이력을 남기기 위함입니다.
                new_prediction_record = Prediction(
                    stock_ticker=stock_ticker,
                    prediction_made_at=now_utc,
                    predicted_value_for_time=predicted_value_for_time,
                    predicted_value=predicted_value,
                    features_json=features_dict,
                    actual_value=actual_value,
                    is_correct=abs(predicted_value - actual_value) / actual_value <= 0.01 if actual_value != 0 else False, # 1% 오차율로 일단 계산
                    error_margin=abs(predicted_value - actual_value) / actual_value if actual_value != 0 else float('inf'),
                    model_version=active_model_entries[model_to_update_type].version_tag
                )
                db.add(new_prediction_record)
                db.commit() # 각 예측마다 커밋하여 실패 시 롤백 방지
                
                if has_financial_data:
                    updates_made_with_financials += 1
                else:
                    updates_made_no_financials += 1

            except Exception as e:
                db.rollback()
                logger.error(f"Failed to learn from data for {stock_ticker} at {predicted_value_for_time}: {e}", exc_info=True)
                # continue to next random stock

        if updates_made_with_financials == 0 and updates_made_no_financials == 0:
            logger.info("No successful updates were made to any model during this run.")
            return

        logger.info(f"Successfully applied {updates_made_with_financials} updates to 'with_financials' model.")
        logger.info(f"Successfully applied {updates_made_no_financials} updates to 'no_financials' model.")

        for model_type, model_instance in models_to_update.items():
            model_entry = active_model_entries.get(model_type)
            if not model_entry:
                logger.warning(f"No active model entry found for {model_type} in DB. Skipping save for this model.")
                continue

            gcs_path_for_type = active_gcs_paths[model_type]
            
            model_entry.update_count_since_last_outdated_save += 1 
            if model_entry.update_count_since_last_outdated_save >= OUTDATED_SAVE_THRESHOLD:
                logger.info(f"Saving outdated model for {model_type} as update count reached threshold.")
                # 현재 GCS에 저장되어 있는 '활성화된' 모델을 백업 (아직 업데이트되지 않은 버전)
                current_active_model_for_outdated_save = active_models[model_type]
                filename_prefix = MODEL_WITH_FINANCIALS_FILENAME.replace(".pkl", "") if model_type == "with_financials" else MODEL_NO_FINANCIALS_FILENAME.replace(".pkl", "")
                save_outdated_model_to_gcs(current_active_model_for_outdated_save, filename_prefix)
                model_entry.update_count_since_last_outdated_save = 0
            
            save_model_to_gcs(model_instance, gcs_path_for_type) # 학습된 새 모델을 GCS에 저장
            
            model_entry.version_tag = f"updated_{model_type}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
            model_entry.trained_at = datetime.datetime.now(datetime.timezone.utc)
            db.add(model_entry)
            db.commit()
            logger.info(f"Active '{model_type}' model version in DB updated and counter handled.")

        logger.info("--- Incremental Model Update Complete ---")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during model update process with random data: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    db = SessionLocal()
    try:
        # --- 기존 초기 모델 버전 설정 로직 유지 ---
        initial_with_financials_entry = db.query(ModelVersion).filter(ModelVersion.model_type == "with_financials", ModelVersion.is_active == True).first()
        if not initial_with_financials_entry:
            if get_gcs_blob(MODEL_WITH_FINANCIALS_GCS_PATH).exists():
                initial_with_financials_entry = ModelVersion(
                    version_tag="initial_with_financials",
                    model_type="with_financials",
                    gcs_path=MODEL_WITH_FINANCIALS_GCS_PATH,
                    performance_metric=None,
                    is_active=True,
                    update_count_since_last_outdated_save=0
                )
                db.add(initial_with_financials_entry)
                db.commit()
                logger.info("'with_financials' initial model version set up.")
            else:
                logger.error(f"Initial model '{MODEL_WITH_FINANCIALS_GCS_PATH}' not found in GCS for 'with_financials'. Please upload it first.")
        
        initial_no_financials_entry = db.query(ModelVersion).filter(ModelVersion.model_type == "no_financials", ModelVersion.is_active == True).first()
        if not initial_no_financials_entry:
            if get_gcs_blob(MODEL_NO_FINANCIALS_GCS_PATH).exists():
                initial_no_financials_entry = ModelVersion(
                    version_tag="initial_no_financials",
                    model_type="no_financials",
                    gcs_path=MODEL_NO_FINANCIALS_GCS_PATH,
                    performance_metric=None,
                    is_active=True,
                    update_count_since_last_outdated_save=0
                )
                db.add(initial_no_financials_entry)
                db.commit()
                logger.info("'no_financials' initial model version set up.")
            else:
                logger.error(f"Initial model '{MODEL_NO_FINANCIALS_GCS_PATH}' not found in GCS for 'no_financials'. Please upload it first.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during initial model setup: {e}", exc_info=True)
    finally:
        db.close() # 초기 설정 후 DB 세션 닫기

    # --- 추가된 부분: 종목 풀 로드 및 모델 업데이트 함수 호출 ---
    load_stock_pool() # 종목 풀 로드
    update_models_with_random_data(num_updates_per_run=1) # 한 번 실행 시 5개 종목에 대해 업데이트 시도 (조정 가능)