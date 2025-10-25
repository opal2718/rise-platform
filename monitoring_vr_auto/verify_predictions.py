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
from typing import Dict, Any, Optional, List

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

# --- MODIFIED: GCS 경로 추가 ---
GCS_NASDAQ_LISTED_FILE = os.path.join(GCS_MODEL_FOLDER, 'nasdaqlisted.txt')
GCS_KRX_LISTED_FILE = os.path.join(GCS_MODEL_FOLDER, 'data_5208_20251025.csv')
# --- END MODIFIED ---

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

_LOADED_MODELS = {} # Cache for models
_LOADED_TICKERS = {} # Cache for tickers list

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

# Dummy function for fetching news/sentiment data (replace with actual implementation)
def fetch_news_sentiment_data(stock_ticker: str, date: datetime.datetime) -> Dict[str, Any]:
    return {
        'stock_sentiment_avg': random.uniform(-0.5, 0.5),
        'stock_news_total_count': random.randint(0, 100),
        'sector_sentiment_avg': random.uniform(-0.3, 0.3),
        'sector_relevance_avg': random.uniform(0, 1),
    }

# Dummy function for fetching financial data (replace with actual implementation)
def fetch_financial_data(stock_ticker: str, date: datetime.datetime) -> Optional[Dict[str, Any]]:
    # Simulate occasional missing financial data (e.g., if a company doesn't report financials for that period)
    if random.random() < 0.3: 
        return None
    
    financial_data = {}
    for col in financial_feature_columns_list:
        if col.endswith('CFS') or col.endswith('OFS'):
            financial_data[col] = random.uniform(-1_000_000_000, 1_000_000_000)
        elif col in ['PBR', 'PER', 'ROR']:
            financial_data[col] = random.uniform(0.1, 50.0)
    return financial_data

# --- MODIFIED: get_all_tradeable_tickers to load from GCS ---
def get_all_tradeable_tickers(nasdaq_gcs_path: str = GCS_NASDAQ_LISTED_FILE, krx_gcs_path: str = GCS_KRX_LISTED_FILE) -> List[str]:
    """
    Loads US (NASDAQ) and KR (KOSPI/KOSDAQ) tickers from GCS files.
    Returns a combined list of tickers, correctly formatted for yfinance.
    Caches the loaded tickers to avoid repeated GCS calls.
    """
    if 'all_tickers' in _LOADED_TICKERS:
        logger.debug("Returning cached ticker list.")
        return _LOADED_TICKERS['all_tickers']

    us_tickers = []
    kr_tickers = []

    # 1. Load US (NASDAQ) tickers from GCS
    try:
        nasdaq_blob = get_gcs_blob(nasdaq_gcs_path)
        if nasdaq_blob.exists():
            with nasdaq_blob.open("r", encoding='utf-8') as f:
                # Skip header line
                header = f.readline()
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) > 0:
                        symbol = parts[0]
                        # Basic filtering for common stocks/ADRs
                        if len(symbol) <= 5 and not symbol.endswith(('W', 'R', 'U', 'P')): # Added 'P' for preferred
                            us_tickers.append(symbol)
            logger.info(f"Loaded {len(us_tickers)} US (NASDAQ) tickers from GCS: {nasdaq_gcs_path}.")
        else:
            logger.error(f"NASDAQ ticker file not found in GCS at {nasdaq_gcs_path}.")
    except Exception as e:
        logger.error(f"Error loading NASDAQ tickers from GCS: {e}")

    # 2. Load KR (KOSPI/KOSDAQ) tickers from GCS
    try:
        krx_blob = get_gcs_blob(krx_gcs_path)
        if krx_blob.exists():
            # Read CSV content into a string buffer, then to pandas
            with krx_blob.open("r", encoding='euc-kr') as f:
                krx_df = pd.read_csv(f, dtype={'단축코드': str})
            
            # Filter out preferred stocks or other non-standard issues
            # '주식종류' column can be used for more robust filtering if needed
            krx_df = krx_df[krx_df['증권구분'] == '주권']
            
            for index, row in krx_df.iterrows():
                ticker = row['단축코드']
                market_type = row['시장구분']
                # Check for NaNs or empty strings in ticker/market_type
                if pd.notna(ticker) and pd.notna(market_type) and ticker.strip() != '':
                    if market_type == 'KOSPI':
                        kr_tickers.append(f"{ticker}.KS")
                    elif market_type in ['KOSDAQ', 'KOSDAQ GLOBAL']:
                        kr_tickers.append(f"{ticker}.KQ")
            logger.info(f"Loaded {len(kr_tickers)} KR (KOSPI/KOSDAQ) tickers from GCS: {krx_gcs_path}.")
        else:
            logger.error(f"KRX ticker file not found in GCS at {krx_gcs_path}.")
    except Exception as e:
        logger.error(f"Error loading KRX tickers from GCS: {e}")

    # Combine both lists
    all_tickers = us_tickers + kr_tickers
    if not all_tickers:
        logger.error("No tickers loaded from GCS. Using a predefined fallback list.")
        fallback_tickers = ['AAPL', 'MSFT', '005930.KS', '000660.KS'] # Fallback
        _LOADED_TICKERS['all_tickers'] = fallback_tickers
        return fallback_tickers
    
    _LOADED_TICKERS['all_tickers'] = all_tickers
    return all_tickers

def run_single_model_training_iteration(tolerance_percent=0.01):
    db = SessionLocal()
    try:
        logger.info("--- Starting Single Model Training Iteration ---")
        now_utc = datetime.datetime.now(datetime.timezone.utc)

        # 1. Randomly select a stock ticker from combined US/KR list
        all_tickers = get_all_tradeable_tickers() # This now loads from GCS
        if not all_tickers:
            logger.error("No tradeable tickers available for selection. Exiting.")
            return

        selected_ticker = random.choice(all_tickers)
        logger.info(f"Randomly selected stock ticker for training: {selected_ticker}")

        # 2. Fetch historical data for the selected ticker to determine valid training dates
        try:
            ticker_obj = yf.Ticker(selected_ticker)
            
            # Fetch daily data for a broader range to pick a random date
            hist_daily = ticker_obj.history(period="max", interval="1d")
            
            if hist_daily.empty:
                logger.warning(f"No historical daily data from yfinance for {selected_ticker}. Skipping.")
                return

            if hist_daily.index.tz is None:
                hist_daily.index = hist_daily.index.tz_localize('UTC')

            # Filter for dates that allow for LAG_PERIODS (90) days of lag data BEFORE the prediction date
            # And also ensure the prediction date is not today or in the future
            if len(hist_daily) < LAG_PERIODS + 1:
                logger.warning(f"Not enough trading days ({len(hist_daily)}) for {selected_ticker} to create {LAG_PERIODS} lag features and a target. Skipping.")
                return

            trading_days = hist_daily.index.sort_values().tolist()

            yesterday_utc = (now_utc - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            valid_trading_days_for_prediction = [d for d in trading_days if d < yesterday_utc]

            if len(valid_trading_days_for_prediction) < LAG_PERIODS + 1:
                logger.warning(f"Not enough past trading days for {selected_ticker} to perform training. Skipping.")
                return

            random_prediction_date_index = random.randint(LAG_PERIODS, len(valid_trading_days_for_prediction) - 1)
            random_prediction_date_ts = valid_trading_days_for_prediction[random_prediction_date_index]
            random_prediction_date = random_prediction_date_ts.to_pydatetime()
            
            logger.info(f"Randomly selected prediction date for {selected_ticker}: {random_prediction_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Extract data for features (LAG_PERIODS days before prediction date)
            features_data_df = hist_daily[
                (hist_daily.index < random_prediction_date_ts) &
                (hist_daily.index >= valid_trading_days_for_prediction[random_prediction_date_index - LAG_PERIODS])
            ].sort_index(ascending=False) # Sort descending to easily get lag_1, lag_2 etc.
            
            if len(features_data_df) < LAG_PERIODS:
                logger.warning(f"Insufficient feature data ({len(features_data_df)} days) for {selected_ticker} for prediction on {random_prediction_date.strftime('%Y-%m-%d')}. Skipping.")
                return

            # The actual value for the prediction date
            actual_value_for_prediction_date_df = hist_daily[
                (hist_daily.index >= random_prediction_date_ts) &
                (hist_daily.index < random_prediction_date_ts + datetime.timedelta(days=1))
            ]

            if actual_value_for_prediction_date_df.empty:
                logger.warning(f"No actual closing price found for {selected_ticker} on {random_prediction_date.strftime('%Y-%m-%d')}. Skipping.")
                return
            
            actual_value = actual_value_for_prediction_date_df['Close'].iloc[0]
            if isinstance(actual_value, (np.float32, np.float64, np.number)):
                actual_value = actual_value.item()
            else:
                actual_value = float(actual_value)

            # Construct features_dict
            features_dict = {}
            
            # Base features (from the last day *before* the prediction date)
            last_day_data = features_data_df.iloc[0] 
            features_dict['Close'] = last_day_data['Close']
            features_dict['Volume'] = last_day_data['Volume']

            # Lag features
            for i in range(1, LAG_PERIODS + 1):
                if i-1 < len(features_data_df): # Ensure index is within bounds
                    lag_data = features_data_df.iloc[i-1] 
                    features_dict[f'Close_lag_{i}'] = lag_data['Close']
                    features_dict[f'Volume_lag_{i}'] = lag_data['Volume']
                else:
                    logger.warning(f"Not enough lag data for Close_lag_{i} for {selected_ticker} for prediction on {random_prediction_date.strftime('%Y-%m-%d')}.")
                    features_dict[f'Close_lag_{i}'] = None
                    features_dict[f'Volume_lag_{i}'] = None

            # News and sentiment data (simulated for the day before prediction)
            news_sentiment = fetch_news_sentiment_data(selected_ticker, features_data_df.index[0].to_pydatetime())
            features_dict.update(news_sentiment)

            # Financial data (simulated for the most recent available quarter/year before prediction)
            financial_data = fetch_financial_data(selected_ticker, features_data_df.index[0].to_pydatetime())
            if financial_data:
                features_dict.update(financial_data)
            else:
                for col in financial_feature_columns_list:
                    features_dict[col] = None 

            # Determine model type
            has_financial_data = determine_financial_data_presence(features_dict)
            model_to_use_type = "with_financials" if has_financial_data else "no_financials"
            
            # Load active models
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

            if model_to_use_type not in active_models:
                logger.warning(f"Active model '{model_to_use_type}' not loaded/available. Skipping training for {selected_ticker} on {random_prediction_date.strftime('%Y-%m-%d')}.")
                return

            # Make a copy of the model for this iteration's learning to avoid modifying the cached model before potential outdated save
            model_for_learning = pickle.loads(pickle.dumps(active_models[model_to_use_type]))

            # Perform prediction (dummy prediction for now, a real model would use model_for_learning.predict(X_for_prediction))
            predicted_value = actual_value * (1 + random.uniform(-0.02, 0.02)) # Simulate a prediction

            # Record prediction and verification
            prediction_record = Prediction(
                stock_ticker=selected_ticker,
                prediction_made_at=now_utc,
                predicted_value_for_time=random_prediction_date,
                predicted_value=predicted_value,
                features_json=features_dict,
                actual_value=actual_value
            )

            error_margin_calculated = abs(prediction_record.predicted_value - prediction_record.actual_value) / prediction_record.actual_value
            if isinstance(error_margin_calculated, (np.float32, np.float64, np.number)):
                prediction_record.error_margin = error_margin_calculated.item()
            else:
                prediction_record.error_margin = float(error_margin_calculated)

            prediction_record.is_correct = prediction_record.error_margin <= tolerance_percent
            db.add(prediction_record)
            db.commit()
            logger.info(f"Recorded and verified simulated prediction {prediction_record.id} for {prediction_record.stock_ticker} @ {prediction_record.predicted_value_for_time.strftime('%Y-%m-%d %H:%M:%S UTC')}: Predicted={prediction_record.predicted_value:.4f}, Actual={prediction_record.actual_value:.4f}, Error={prediction_record.error_margin:.2%}, Correct={prediction_record.is_correct}")

            # Incremental Model Update based on this single record
            logger.info(f"Applying incremental update to '{model_to_use_type}' model with the new verified data.")
            
            selected_feature_cols = full_feature_columns if has_financial_data else no_financial_feature_columns
            
            X_for_learn = {}
            data_is_complete = True
            for col in selected_feature_cols:
                value = features_dict.get(col)
                if value is None:
                    logger.warning(f"Missing or None feature '{col}' for the selected prediction record. Skipping learning for this record.")
                    data_is_complete = False
                    break
                X_for_learn[col] = value
            
            if data_is_complete:
                model_to_update_instance = model_for_learning # Use the copy we made
                model_to_update_instance.learn_one(X_for_learn, actual_value)
                logger.info(f"Model '{model_to_use_type}' has learned from the latest data.")

                # Update model version in DB and GCS
                model_entry = active_model_entries.get(model_to_use_type)
                if model_entry:
                    model_entry.update_count_since_last_outdated_save += 1 
                    if model_entry.update_count_since_last_outdated_save >= OUTDATED_SAVE_THRESHOLD:
                        logger.info(f"Saving outdated model for {model_to_use_type} as update count reached threshold.")
                        # Load the *currently active* model from cache/GCS to save as outdated
                        current_active_model_for_outdated_save = active_models[model_to_use_type] 
                        filename_prefix = MODEL_WITH_FINANCIALS_FILENAME.replace(".pkl", "") if model_to_use_type == "with_financials" else MODEL_NO_FINANCIALS_FILENAME.replace(".pkl", "")
                        save_outdated_model_to_gcs(current_active_model_for_outdated_save, filename_prefix)
                        model_entry.update_count_since_last_outdated_save = 0
                    
                    save_model_to_gcs(model_to_update_instance, active_gcs_paths[model_to_use_type])
                    
                    model_entry.version_tag = f"updated_{model_to_use_type}_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
                    model_entry.trained_at = datetime.datetime.now(datetime.timezone.utc)
                    db.add(model_entry)
                    db.commit()
                    logger.info(f"Active '{model_to_use_type}' model version in DB updated and counter handled.")
                else:
                    logger.warning(f"No active model entry found for {model_to_use_type} in DB. Skipping DB update and GCS save for this model.")
            else:
                logger.warning(f"Skipped learning for {selected_ticker} on {random_prediction_date.strftime('%Y-%m-%d')} due to incomplete features.")

        except Exception as e:
            logger.error(f"Error during single training iteration for {selected_ticker}: {e}", exc_info=True)

    except Exception as e:
        db.rollback()
        logger.error(f"General error during run_single_model_training_iteration: {e}", exc_info=True)
    finally:
        db.close()
        logger.info("--- Single Model Training Iteration Complete ---")


if __name__ == "__main__":
    db = SessionLocal()
    try:
        # Initial setup for "with_financials" model
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
        
        # Initial setup for "no_financials" model
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

    # Run a single training iteration
    run_single_model_training_iteration()