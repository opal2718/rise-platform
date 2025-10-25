import os
from fastapi import FastAPI, Request, HTTPException
import httpx
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pickle
import pandas as pd
import numpy as np # numpy import
import logging
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import uuid
import json

#--- River library components (MUST be present for model deserialization) ---
# FIX: Import ARFRegressor from river.forest
from river.forest import ARFRegressor
from river import preprocessing # preprocessing.StandardScaler()를 위해 필요

# =========================================================================
# --- 중요: Log1pTransformer 클래스를 여기에 다시 포함시켜야 합니다. ---
# pickle.ize된 모델이 이 클래스를 찾을 수 있도록 합니다.
# =========================================================================

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
# =========================================================================

import sys
# === CRITICAL FIX: Ensure Log1pTransformer is available in the __main__ module for pickling ===
# When the model was pickled, if the training script was run directly,
# Log1pTransformer was in the '__main__' module.
# When uvicorn runs this app, this file becomes '__main__'.
# So, we explicitly ensure the class is set in sys.modules['__main__']
# for pickle to find it correctly during deserialization.
sys.modules['__main__'].Log1pTransformer = Log1pTransformer


#--- Database setup (unchanged) ---

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://postgres:wegoUP1234!@34.45.237.0:5432/postgres")
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions" # Fixed: tablename should be __tablename__
    id = Column(Integer, primary_key=True)
    stock_ticker = Column(String(10), nullable=False)
    prediction_made_at = Column(DateTime(timezone=True), default=datetime.datetime.now(datetime.timezone.utc))
    predicted_value_for_time = Column(DateTime(timezone=True), nullable=False)
    predicted_value = Column(Float, nullable=False)
    features_json = Column(JSONB, nullable=False)
    actual_value = Column(Float)
    is_correct = Column(Boolean)
    error_margin = Column(Float)
    model_version = Column(String(50), default='initial')

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) # Use __name__ for the logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For testing, allow all origins
    allow_methods=["*"], # For testing, allow all methods
    allow_headers=["*"],
)

# Global configuration

STOCK_PERIOD = "1d"
LAG_PERIODS = 90

# GCS 마운트 경로 (실제 배포 환경에서 모델 파일이 저장된 경로)

GCS_PATH = "/mnt/gcs"

#--- NEW: 두 가지 모델 파일 경로 정의 ---

MODEL_WITH_FINANCIALS_FILENAME = "model_with_financials.pkl"
MODEL_NO_FINANCIALS_FILENAME = "model_no_financials.pkl"
MODEL_WITH_FINANCIALS_FILEPATH = os.path.join(GCS_PATH, MODEL_WITH_FINANCIALS_FILENAME)
MODEL_NO_FINANCIALS_FILEPATH = os.path.join(GCS_PATH, MODEL_NO_FINANCIALS_FILENAME)

#--- NEW: feature_columns 정의 (훈련 스크립트와 동일하게) ---

base_feature_columns = ['Close', 'Volume', 'stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']
lagged_base_features = [f'{col}_lag_{i}' for col in ['Close', 'Volume'] for i in range(1, LAG_PERIODS + 1)] # Changed _lag_ to match training script

financial_feature_columns_list = [
    '유동자산CFS', '비유동자산CFS', '자산총계CFS', '유동부채CFS', '비유동부채CFS',
    '부채총계CFS', '자본금CFS', '이익잉여금CFS', '자본총계CFS', '매출액CFS',
    '영업이익CFS', '법인세차감전 순이익CFS', '당기순이익(손실)CFS', '총포괄손익CFS',
    '유동자산OFS', '비유동자산OFS', '자산총계OFS', '유동부채OFS', '비유동부채OFS',
    '부채총계OFS', '자본금OFS', '이익잉여금OFS', '자본총계OFS', '매출액OFS',
    '영업이익OFS', '법인세차감전 순이익OFS', '당기순이익(손실)OFS', '총포괄손익OFS',
    'PBR', 'PER', 'ROR'
]
full_feature_columns = base_feature_columns + lagged_base_features + financial_feature_columns_list
no_financial_feature_columns = base_feature_columns + lagged_base_features

#--- Helper function to load the model (MODIFIED to load specific model) ---

def load_model(model_type: str):
    filepath = ""
    if model_type == "with_financials":
        filepath = MODEL_WITH_FINANCIALS_FILEPATH
    elif model_type == "no_financials":
        filepath = MODEL_NO_FINANCIALS_FILEPATH
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'with_financials' or 'no_financials'.")

    if not os.path.exists(filepath):
        logger.error(f"Model file not found at: {filepath}. Please ensure models are correctly mounted/available.")
        raise FileNotFoundError(f"Model file not found at: {filepath}. Please ensure models are correctly mounted/available.")
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Model '{model_type}' loaded successfully from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load model '{model_type}': {e}")
#--- REVISED /predict endpoint (MODIFIED with None to 0.0 conversion) ---

@app.post("/predict")
async def predict(data: dict, request: Request):
    logger.info(f"Received /predict request with data: {data}")

    if "text" not in data:
        logger.error("Missing 'text' field in /predict request data.")
        raise HTTPException(status_code=400, detail="Missing 'text' field in request body")

    stock = data["text"]

    my_url = str(request.base_url).rstrip("/")
    # TODO: "http://ai2:8001/data"는 실제 ai2 서비스의 엔드포인트여야 합니다.
    data_url = f"http://ai2:8001/data?stock={stock}&period={STOCK_PERIOD}"
    logger.info(f"Fetching *single instance* processed data from internal URL: {data_url}")

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(data_url, timeout=30.0)
            resp.raise_for_status()
            request_data = resp.json()
        logger.info(f"Successfully fetched processed data for stock {stock} from /data.")
    except httpx.TimeoutException:
        logger.error(f"Timeout when fetching data from {data_url}")
        raise HTTPException(status_code=504, detail=f"Data fetching from internal service timed out.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Error status {e.response.status_code} when fetching data from {data_url}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Failed to fetch processed data from internal service: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Network error when fetching data from {data_url}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Could not connect to internal data service: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during data fetching from {data_url}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching data: {e}")

    try:
        processed_X_for_prediction = request_data.get('processed_features_for_prediction')
        latest_timestamp_str = request_data.get('latest_timestamp')

        if not processed_X_for_prediction or not latest_timestamp_str:
            logger.error("Invalid or missing 'processed_features_for_prediction' or 'latest_timestamp' from /data endpoint.")
            raise HTTPException(status_code=422, detail="Invalid processed features format available for prediction.")

        last_timestamp_of_features = pd.to_datetime(latest_timestamp_str)
        prediction_target_time = last_timestamp_of_features + pd.Timedelta(days=1)

        logger.info(f"Processed single instance for model (raw from /data): {processed_X_for_prediction}")
        logger.info(f"Prediction target time based on features: {prediction_target_time}")

        # --- NEW LOGIC: Check for financial data presence ---
        # is_korean_stock 정보는 /data 엔드포인트에서 받아오지 못하므로, stock_ticker로 판단
        is_korean_stock = stock.endswith(".KS")
        
        has_financial_data = False
        if is_korean_stock: # 한국 주식일 경우에만 재무 데이터 존재 가능성 확인
            # financial_feature_columns_list에 있는 모든 피처가 None이 아닌지 확인
            all_financial_features_present = True
            for financial_col in financial_feature_columns_list:
                if processed_X_for_prediction.get(financial_col) is None:
                    all_financial_features_present = False
                    break
            
            if all_financial_features_present:
                has_financial_data = True
                logger.info(f"All financial features seem to be present for stock {stock}. Will consider 'with_financials' model.")
            else:
                logger.warning(f"Some financial features are missing (None/NaN) for stock {stock}. Will use 'no_financials' model.")
        else:
            logger.info(f"Stock {stock} is not a Korean stock. Will use 'no_financials' model.")


        selected_model_type = "with_financials" if has_financial_data else "no_financials"
        selected_feature_columns = full_feature_columns if has_financial_data else no_financial_feature_columns
        logger.info(f"Selected model type: {selected_model_type} for stock {stock}")

        # 선택된 feature_columns에 따라 입력 데이터 준비 및 None(NaN)을 0.0으로 변환
        final_X_for_model = {}
        for col in selected_feature_columns:
            value = processed_X_for_prediction.get(col)
            # 학습 시와 동일하게 None(NaN) 값을 0.0으로 변환하여 모델에 전달
            if value is None: # JSON에서 np.nan은 None으로 직렬화됩니다.
                final_X_for_model[col] = 0.0
            else:
                final_X_for_model[col] = float(value) # float로 명시적 변환

        logger.info(f"Prepared final_X_for_model for prediction (None to 0.0 converted): {final_X_for_model}")

    except Exception as e:
        logger.error(f"Error during final data preparation in /predict for stock {stock}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Data preparation in /predict failed: {e}")

# 5. Model Loading and Prediction
    try:
        model = load_model(selected_model_type) # 동적으로 모델 로드

        # model.predict_one() 메소드가 딕셔너리 입력을 받아들이고 float 값을 반환한다고 가정합니다.
        prediction_result = model.predict_one(final_X_for_model)
        predicted_value = float(prediction_result)

        logger.info(f"⚙️ Predicted value type: {type(predicted_value)}")
        logger.info(f"⚙️ Predicted value: {predicted_value:.2f}")
        logger.info(f"Prediction made successfully for stock {stock} using {selected_model_type} model: {predicted_value:.2f} for {prediction_target_time}")

        # Prepare features for storage (store the single instance used for prediction)
        features_to_store = processed_X_for_prediction

        # Store prediction in database
        db = SessionLocal()
        try:
            new_prediction = Prediction(
                stock_ticker=stock,
                prediction_made_at=datetime.datetime.now(datetime.timezone.utc),
                predicted_value_for_time=prediction_target_time.to_pydatetime().replace(tzinfo=datetime.timezone.utc),
                predicted_value=predicted_value,
                features_json=features_to_store, # 원본 데이터를 JSON으로 저장
                model_version=selected_model_type # 사용된 모델의 버전을 저장
            )
            db.add(new_prediction)
            db.commit()
            db.refresh(new_prediction)
            logger.info(f"Prediction {new_prediction.id} stored in database with features using model version: {selected_model_type}.")
        except Exception as db_e:
            db.rollback()
            logger.error(f"Failed to store prediction in database: {db_e}", exc_info=True)
        finally:
            db.close()

        return {"prediction": predicted_value}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except AttributeError as e:
        logger.error(f"Model error for stock {stock} (AttributeError - likely model.predict_one issue or method mismatch): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model operation failed, check model's 'predict_one' method: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during model prediction for stock {stock}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during prediction: {e}")

@app.post("/process")
async def process(data: dict):
    logger.info(f"Received /process request with data: {data}")
    A = data
    B = A
    logger.info("Process endpoint received data: %s", data)
    return {"output": B}

@app.post("/hello")
async def hello():
    logger.info("Received /hello request.")
    return {"message": f"Hello from ai1 service!"}