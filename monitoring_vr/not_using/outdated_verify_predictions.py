# update_model.py
import os
import datetime
import pandas as pd
import yfinance as yf # Still needed for getting historical data for "actuals" if not found in DB
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB # For JSONB column
import pickle
from google.cloud import storage
import logging
import json # For loading JSON features
import numpy as np # For NaN check

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
    features_json = Column(JSONB, nullable=False) # Features stored as JSONB
    actual_value = Column(Float)
    is_correct = Column(Boolean)
    error_margin = Column(Float)
    model_version = Column(String(50)) # "with_financials" or "no_financials"

class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    version_tag = Column(String(50), unique=True, nullable=False) # "initial_with_financials", "updated_with_financials_2023..."
    model_type = Column(String(50), nullable=False) # "with_financials" or "no_financials"
    trained_at = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc))
    gcs_path = Column(String(255), nullable=False)
    performance_metric = Column(Float) # e.g., RMSE on validation set
    is_active = Column(Boolean, default=False)

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# GCS configuration
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-gcs-bucket-name")
GCS_MODEL_FOLDER = "models"
MODEL_FILENAME_PREFIX = "model_"
MODEL_PICKLE_EXT = ".pkl"

# --- NEW: 두 모델의 GCS 파일명 및 경로 정의 ---
MODEL_WITH_FINANCIALS_FILENAME = "model_with_financials.pkl"
MODEL_NO_FINANCIALS_FILENAME = "model_no_financials.pkl"
MODEL_WITH_FINANCIALS_GCS_PATH = os.path.join(GCS_MODEL_FOLDER, MODEL_WITH_FINANCIALS_FILENAME)
MODEL_NO_FINANCIALS_GCS_PATH = os.path.join(GCS_MODEL_FOLDER, MODEL_NO_FINANCIALS_FILENAME)

# --- NEW: feature_columns 정의 (ai1과 동일하게 유지) ---
LAG_PERIODS = 90 # ai1과 동일하게 유지
# 재무제표 피처가 포함된 전체 피처 목록
full_feature_columns = ['Close', 'Volume', 'stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']
for i in range(1, LAG_PERIODS + 1):
    full_feature_columns.append(f'Close_lag_{i}')
    full_feature_columns.append(f'Volume_lag_{i}')

# 재무제표 피처만
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

# 재무제표 피처가 없는 모델을 위한 피처 목록
no_financial_feature_columns = [col for col in full_feature_columns if col not in financial_feature_columns_list]

# 전역적으로 모델을 캐시하기 위한 딕셔너리
_LOADED_MODELS = {}

def get_gcs_blob(filepath):
    client = storage.Client()
    bucket = client.get_bucket(GCS_BUCKET_NAME)
    return bucket.blob(filepath)

def load_model_from_gcs(filepath):
    """Loads a model from GCS and caches it."""
    if filepath in _LOADED_MODELS:
        logger.debug(f"Model already in cache: {filepath}")
        return _LOADED_MODELS[filepath]

    blob = get_gcs_blob(filepath)
    if not blob.exists():
        logger.error(f"Model file not found in GCS at: {filepath}")
        raise FileNotFoundError(f"Model file not found in GCS at: {filepath}")
    
    with blob.open("rb") as f:
        model = pickle.load(f)
    _LOADED_MODELS[filepath] = model # Cache the loaded model
    logger.info(f"Model loaded successfully from GCS: {filepath}")
    return model

def save_model_to_gcs(model, filepath):
    blob = get_gcs_blob(filepath)
    with blob.open("wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved successfully to GCS: {filepath}")
    # Cache도 업데이트
    _LOADED_MODELS[filepath] = model

def determine_financial_data_presence(features_dict: dict) -> bool:
    """
    Given a features dictionary, determines if financial data features are present (not NaN/None).
    """
    for financial_col in financial_feature_columns_list:
        # ai1에서 None으로 변환된 NaN을 확인
        if features_dict.get(financial_col) is None:
            return False
    return True

def incremental_model_update(lookback_days=7):
    db = SessionLocal()
    try:
        # 1. Load the currently active models (both types)
        # ModelVersion 테이블에서 현재 활성 모델 정보를 가져옵니다.
        active_models_info = db.query(ModelVersion).filter(ModelVersion.is_active == True).all()
        
        active_models = {}
        active_gcs_paths = {}
        for info in active_models_info:
            active_models[info.model_type] = load_model_from_gcs(info.gcs_path)
            active_gcs_paths[info.model_type] = info.gcs_path
        
        if not active_models:
            logger.error("No active models found in ModelVersion table. Cannot perform update.")
            return

        # 2. Query for recently verified predictions
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(days=lookback_days)

        recent_verified_predictions = db.query(Prediction).filter(
            Prediction.actual_value.isnot(None),
            Prediction.predicted_value_for_time >= start_date,
            Prediction.predicted_value_for_time <= end_date
        ).all()

        if not recent_verified_predictions:
            logger.info(f"No recent verified predictions in the last {lookback_days} days to update the models.")
            return

        logger.info(f"Found {len(recent_verified_predictions)} verified predictions for incremental update.")

        updates_made_with_financials = 0
        updates_made_no_financials = 0

        # 3. Iterate and call model.learn_one() based on financial data presence
        for prediction_record in recent_verified_predictions:
            try:
                features_dict = prediction_record.features_json
                actual_target = prediction_record.actual_value

                # 재무 데이터 유무 판단
                has_financial_data = determine_financial_data_presence(features_dict)
                
                model_to_update_type = "with_financials" if has_financial_data else "no_financials"
                
                # 해당 모델이 현재 로드되어 있는지 확인
                if model_to_update_type not in active_models:
                    logger.warning(f"Active model '{model_to_update_type}' not loaded. Skipping update for prediction {prediction_record.id}.")
                    continue

                selected_model = active_models[model_to_update_type]
                
                # 모델에 맞는 피처 목록 선택
                if has_financial_data:
                    selected_feature_cols = full_feature_columns
                else:
                    selected_feature_cols = no_financial_feature_columns
                
                # 모델의 learn_one 메소드가 기대하는 형식으로 피처 데이터 준비
                # ai1에서 JSONB로 저장된 값은 None으로 유지될 수 있습니다.
                # 모델의 learn_one이 None을 처리할 수 있어야 합니다.
                # 만약 scikit-learn 모델이라면, 여기서 np.nan으로 변환 후 Imputation 필요
                # 예: X_for_learn = pd.DataFrame([{col: features_dict.get(col, np.nan) for col in selected_feature_cols}])
                # target_for_learn = pd.Series([actual_target])
                # selected_model.learn_one(X_for_learn, target_for_learn)

                # 현재는 모델이 딕셔너리 입력을 처리할 수 있다고 가정
                X_for_learn = {col: features_dict.get(col) for col in selected_feature_cols}


                logger.debug(f"Learning from prediction {prediction_record.id} for stock {prediction_record.stock_ticker} using {model_to_update_type} model.")
                selected_model.learn_one(X_for_learn, actual_target) # Assuming learn_one accepts dict and scalar

                if has_financial_data:
                    updates_made_with_financials += 1
                else:
                    updates_made_no_financials += 1

            except Exception as e:
                logger.error(f"Failed to learn from prediction {prediction_record.id}: {e}", exc_info=True)
        
        if updates_made_with_financials == 0 and updates_made_no_financials == 0:
            logger.info("No successful updates were made to any model during this run.")
            return

        logger.info(f"Successfully applied {updates_made_with_financials} updates to 'with_financials' model.")
        logger.info(f"Successfully applied {updates_made_no_financials} updates to 'no_financials' model.")

        # 4. Save the updated models and update ModelVersion entries
        for model_type, model_instance in active_models.items():
            gcs_path_for_type = active_gcs_paths[model_type]
            
            # Save (override) the updated model to its existing GCS path
            save_model_to_gcs(model_instance, gcs_path_for_type)
            
            # Update ModelVersion entry
            model_entry = db.query(ModelVersion).filter(
                ModelVersion.is_active == True,
                ModelVersion.model_type == model_type
            ).first()

            if model_entry:
                model_entry.version_tag = f"updated_{model_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                model_entry.trained_at = datetime.datetime.now(datetime.timezone.utc)
                db.add(model_entry)
                db.commit()
                logger.info(f"Active '{model_type}' model version in DB updated after incremental learning.")
            else:
                logger.warning(f"Could not find active '{model_type}' model entry in DB to update. This should not happen if initial setup is correct.")
                # This could be a critical error; consider re-initialization logic

        logger.info("All active models updated and saved to GCS.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during incremental model update process: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    db = SessionLocal()
    try:
        # --- Initial setup for both active models if they don't exist ---
        
        # Setup 'with_financials' model
        initial_with_financials_entry = db.query(ModelVersion).filter(ModelVersion.model_type == "with_financials", ModelVersion.is_active == True).first()
        if not initial_with_financials_entry:
            if get_gcs_blob(MODEL_WITH_FINANCIALS_GCS_PATH).exists():
                initial_with_financials_entry = ModelVersion(
                    version_tag="initial_with_financials",
                    model_type="with_financials",
                    gcs_path=MODEL_WITH_FINANCIALS_GCS_PATH,
                    performance_metric=None,
                    is_active=True
                )
                db.add(initial_with_financials_entry)
                db.commit()
                logger.info("'with_financials' initial model version set up.")
            else:
                logger.error(f"Initial model '{MODEL_WITH_FINANCIALS_GCS_PATH}' not found in GCS for 'with_financials'. Please upload it first.")
        
        # Setup 'no_financials' model
        initial_no_financials_entry = db.query(ModelVersion).filter(ModelVersion.model_type == "no_financials", ModelVersion.is_active == True).first()
        if not initial_no_financials_entry:
            if get_gcs_blob(MODEL_NO_FINANCIALS_GCS_PATH).exists():
                initial_no_financials_entry = ModelVersion(
                    version_tag="initial_no_financials",
                    model_type="no_financials",
                    gcs_path=MODEL_NO_FINANCIALS_GCS_PATH,
                    performance_metric=None,
                    is_active=True
                )
                db.add(initial_no_financials_entry)
                db.commit()
                logger.info("'no_financials' initial model version set up.")
            else:
                logger.error(f"Initial model '{MODEL_NO_FINANCIALS_GCS_PATH}' not found in GCS for 'no_financials'. Please upload it first.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during initial model setup for incremental_model_update: {e}", exc_info=True)
    finally:
        db.close()

    # Run the incremental update
    incremental_model_update()