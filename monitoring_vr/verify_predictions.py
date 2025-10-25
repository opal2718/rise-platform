# verify_predictions.py
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

# --- MODIFIED: GCS_MODEL_FOLDER is now an empty string, meaning root of the bucket ---
GCS_MODEL_FOLDER = "" 
# --- MODIFIED: outdated folder is now directly under the bucket root ---
GCS_OUTDATED_FOLDER = "outdated" # Assuming 'outdated' folder is at the bucket root level

# --- GCS 파일명 및 경로 정의 ---
MODEL_WITH_FINANCIALS_FILENAME = "model_with_financials.pkl"
MODEL_NO_FINANCIALS_FILENAME = "model_no_financials.pkl"
# --- MODIFIED: Paths now directly refer to the bucket root or 'outdated' folder ---
MODEL_WITH_FINANCIALS_GCS_PATH = os.path.join(GCS_MODEL_FOLDER, MODEL_WITH_FINANCIALS_FILENAME) # This will be just "model_with_financials.pkl"
MODEL_NO_FINANCIALS_GCS_PATH = os.path.join(GCS_MODEL_FOLDER, MODEL_NO_FINANCIALS_FILENAME)     # This will be just "model_no_financials.pkl"

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

OUTDATED_SAVE_THRESHOLD = 1

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
    # --- MODIFIED: Outdated files are saved directly under the 'outdated' folder at bucket root ---
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

def verify_and_update_predictions(tolerance_percent=0.01, lookback_days_for_update=7):
    db = SessionLocal()
    try:
        # --- PART 1: Verify Predictions ---
        logger.info("--- Starting Prediction Verification ---")
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        pending_predictions = db.query(Prediction).filter(
            Prediction.actual_value.is_(None),
            Prediction.predicted_value_for_time <= now_utc
        ).all()

        if not pending_predictions:
            logger.info("No pending predictions to verify.")
            pass 
        else:
            logger.info(f"Found {len(pending_predictions)} pending predictions to verify.")

            stocks_to_fetch = set(p.stock_ticker for p in pending_predictions)
            
            fetched_data = {}
            for stock_ticker in stocks_to_fetch:
                try:
                    ticker = yf.Ticker(stock_ticker)
                    hist = ticker.history(period="60d", interval="1h") 
                    if not hist.empty:
                        if hist.index.tz is None:
                            hist.index = hist.index.tz_localize('UTC') 
                        fetched_data[stock_ticker] = hist['Close']
                    else:
                        logger.warning(f"No historical data from yfinance for {stock_ticker}.")
                except Exception as e:
                    logger.error(f"Error fetching yfinance data for {stock_ticker}: {e}")

            for prediction in pending_predictions:
                stock_data = fetched_data.get(prediction.stock_ticker)
                if stock_data is None:
                    logger.warning(f"Skipping prediction {prediction.id}: No historical data for {prediction.stock_ticker}.")
                    continue

                target_time = prediction.predicted_value_for_time
                try:
                    potential_actuals = stock_data[stock_data.index >= target_time].sort_index()
                    if not potential_actuals.empty:
                        actual_value = potential_actuals.iloc[0]
                    else:
                        logger.warning(f"Could not find actual value for {prediction.stock_ticker} at {target_time}.")
                        actual_value = None

                except KeyError:
                    actual_value = None
                
                if actual_value is not None:
                    if isinstance(actual_value, (np.float32, np.float64, np.number)):
                        actual_value_python = actual_value.item()
                    else:
                        actual_value_python = float(actual_value)

                    prediction.actual_value = actual_value_python
                    
                    error_margin_calculated = abs(prediction.predicted_value - actual_value_python) / actual_value_python
                    if isinstance(error_margin_calculated, (np.float32, np.float64, np.number)):
                        prediction.error_margin = error_margin_calculated.item()
                    else:
                        prediction.error_margin = float(error_margin_calculated)

                    prediction.is_correct = prediction.error_margin <= tolerance_percent
                    db.add(prediction)
                    logger.info(f"Verified prediction {prediction.id} for {prediction.stock_ticker} @ {prediction.predicted_value_for_time}: Predicted={prediction.predicted_value:.4f}, Actual={actual_value_python:.4f}, Error={prediction.error_margin:.2%}, Correct={prediction.is_correct}")
                else:
                    logger.warning(f"Failed to find actual value for prediction {prediction.id} (Stock: {prediction.stock_ticker}, Time: {prediction.predicted_value_for_time}). Keeping actual_value as NULL.")

            db.commit()
            logger.info("--- Prediction Verification Complete ---")


        # --- PART 2: Model Update after Verification ---
        logger.info("--- Starting Incremental Model Update ---")

        active_models_info = db.query(ModelVersion).filter(ModelVersion.is_active == True).all()
        
        active_models = {}
        active_gcs_paths = {}
        active_model_entries = {}
        for info in active_models_info:
            try:
                active_models[info.model_type] = load_model_from_gcs(info.gcs_path)
                active_gcs_paths[info.model_type] = info.gcs_path
                active_model_entries[info.model_type] = info
            except FileNotFoundError:
                logger.error(f"Active model file not found in GCS for {info.model_type} at {info.gcs_path}. Skipping update for this model type.")
                continue
        
        if not active_models:
            logger.error("No active models successfully loaded. Cannot perform update.")
            return

        start_date = now_utc - datetime.timedelta(days=lookback_days_for_update)

        recent_verified_predictions = db.query(Prediction).filter(
            Prediction.actual_value.isnot(None),
            Prediction.predicted_value_for_time >= start_date,
            Prediction.predicted_value_for_time <= now_utc
        ).all()

        if not recent_verified_predictions:
            logger.info(f"No recent verified predictions in the last {lookback_days_for_update} days to update the models.")
            return

        logger.info(f"Found {len(recent_verified_predictions)} verified predictions for incremental update.")

        updates_made_with_financials = 0
        updates_made_no_financials = 0
        
        models_to_update = {
            m_type: pickle.loads(pickle.dumps(model_instance))
            for m_type, model_instance in active_models.items()
        }

        for prediction_record in recent_verified_predictions:
            try:
                features_dict = prediction_record.features_json
                actual_target = prediction_record.actual_value

                has_financial_data = determine_financial_data_presence(features_dict)
                
                model_to_update_type = "with_financials" if has_financial_data else "no_financials"
                
                if model_to_update_type not in models_to_update:
                    logger.warning(f"Active model '{model_to_update_type}' not loaded/available. Skipping update for prediction {prediction_record.id}.")
                    continue

                selected_model = models_to_update[model_to_update_type]
                
                if has_financial_data:
                    selected_feature_cols = full_feature_columns
                else:
                    selected_feature_cols = no_financial_feature_columns
                
                # --- BUG PREVENTION: Check if all required features exist and are not None ---
                X_for_learn = {}
                data_is_complete = True
                for col in selected_feature_cols:
                    value = features_dict.get(col)
                    if value is None: # Check for None (including explicit None and missing keys)
                        logger.warning(f"Missing or None feature '{col}' for prediction {prediction_record.id}. Skipping learning for this record.")
                        data_is_complete = False
                        break
                    X_for_learn[col] = value
                
                if not data_is_complete:
                    continue # Skip to the next prediction record if data is incomplete

                logger.debug(f"Learning from prediction {prediction_record.id} for stock {prediction_record.stock_ticker} using {model_to_update_type} model.")
                selected_model.learn_one(X_for_learn, actual_target)

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

        for model_type, model_instance in models_to_update.items():
            model_entry = active_model_entries.get(model_type)
            if not model_entry:
                logger.warning(f"No active model entry found for {model_type} in DB. Skipping save for this model.")
                continue

            gcs_path_for_type = active_gcs_paths[model_type]
            
            model_entry.update_count_since_last_outdated_save += 1 
            if model_entry.update_count_since_last_outdated_save >= OUTDATED_SAVE_THRESHOLD:
                logger.info(f"Saving outdated model for {model_type} as update count reached threshold.")
                current_active_model_for_outdated_save = active_models[model_type]
                filename_prefix = MODEL_WITH_FINANCIALS_FILENAME.replace(".pkl", "") if model_type == "with_financials" else MODEL_NO_FINANCIALS_FILENAME.replace(".pkl", "")
                save_outdated_model_to_gcs(current_active_model_for_outdated_save, filename_prefix)
                model_entry.update_count_since_last_outdated_save = 0
            
            save_model_to_gcs(model_instance, gcs_path_for_type)
            
            model_entry.version_tag = f"updated_{model_type}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
            model_entry.trained_at = datetime.datetime.now(datetime.timezone.utc)
            db.add(model_entry)
            db.commit()
            logger.info(f"Active '{model_type}' model version in DB updated and counter handled.")

        logger.info("--- Incremental Model Update Complete ---")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during verification and model update process: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    db = SessionLocal()
    try:
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
        db.close()

    verify_and_update_predictions()