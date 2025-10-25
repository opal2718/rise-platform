# verify_predictions.py
import os
import datetime
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging
import numpy as np # numpy import 추가 (만약을 대비해)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup (same as in app.py for consistency)
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+psycopg2://user:password@localhost/stock_predictions")
Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    stock_ticker = Column(String(10), nullable=False)
    prediction_made_at = Column(DateTime(timezone=True))
    predicted_value_for_time = Column(DateTime(timezone=True), nullable=False)
    predicted_value = Column(Float, nullable=False)
    actual_value = Column(Float)
    is_correct = Column(Boolean)
    error_margin = Column(Float)
    model_version = Column(String(50))

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def verify_predictions(tolerance_percent=0.01): # 1% tolerance for "correct"
    db = SessionLocal()
    try:
        # Get predictions whose target time has passed and actual_value is null
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        pending_predictions = db.query(Prediction).filter(
            Prediction.actual_value.is_(None),
            Prediction.predicted_value_for_time <= now_utc
        ).all()

        if not pending_predictions:
            logger.info("No pending predictions to verify.")
            return

        logger.info(f"Found {len(pending_predictions)} pending predictions to verify.")

        # Group by stock_ticker to fetch yfinance data efficiently
        stocks_to_fetch = set(p.stock_ticker for p in pending_predictions)
        
        fetched_data = {} # {stock_ticker: yfinance_dataframe}
        for stock_ticker in stocks_to_fetch:
            try:
                # Fetch data covering the period needed for these predictions
                # You might need a more sophisticated range if predictions span wide times
                # For hourly, fetching a few days to cover historical predictions should be enough.
                ticker = yf.Ticker(stock_ticker)
                # Max range for 1h is usually 60 days
                hist = ticker.history(period="60d", interval="1h") 
                if not hist.empty:
                    # Ensure index is timezone-aware if yfinance returns it
                    if hist.index.tz is None:
                        hist.index = hist.index.tz_localize('UTC') # Assume UTC if not specified
                    fetched_data[stock_ticker] = hist['Close'] # Only need 'Close' for verification
                else:
                    logger.warning(f"No historical data from yfinance for {stock_ticker}.")
            except Exception as e:
                logger.error(f"Error fetching yfinance data for {stock_ticker}: {e}")

        for prediction in pending_predictions:
            stock_data = fetched_data.get(prediction.stock_ticker)
            if stock_data is None:
                logger.warning(f"Skipping prediction {prediction.id}: No historical data for {prediction.stock_ticker}.")
                continue

            # Find the actual close price at or immediately after the predicted_value_for_time
            # This requires careful alignment. yfinance '1h' interval timestamps are *start* of the hour.
            # If your model predicts the 'Close' at 10:00, that corresponds to the 09:00 interval in yfinance.
            # You might need to adjust prediction.predicted_value_for_time to match yfinance's index logic.
            
            # For simplicity, let's find the closest timestamp in yfinance data
            # You might need to round predicted_value_for_time to the nearest hour if your predictions are hourly
            target_time = prediction.predicted_value_for_time
            
            # Find the index in stock_data closest to or exactly at target_time
            # This is a heuristic and might need fine-tuning based on yfinance exact timestamping and your prediction horizon
            try:
                # Use asof() for exact match or previous, or reindex for interpolation
                # actual_close_series = stock_data.reindex(index=stock_data.index.union([target_time])).sort_index().loc[target_time] # 이 부분은 사용되지 않아 주석 처리함.

                # If target_time exactly matches an index
                if target_time in stock_data.index:
                    actual_value = stock_data.loc[target_time]
                # If target_time falls between two data points, take the next one or interpolate
                # This depends on what "value for time" precisely means for your model.
                # For now, let's find the nearest point by exact matching after some timestamp adjustment
                else:
                    # Let's try finding the interval that *contains* the target time
                    # Or the interval *ending* at the target time
                    # This is tricky without knowing exact YF timestamp semantics for '1h' close
                    # For example, if YF '1h' 9:30-10:30 is indexed at 9:30, then 10:30 close is 9:30 row.
                    
                    # Simpler: find the next available data point after prediction_target_time
                    potential_actuals = stock_data[stock_data.index >= target_time].sort_index()
                    if not potential_actuals.empty:
                        actual_value = potential_actuals.iloc[0]
                    else:
                        logger.warning(f"Could not find actual value for {prediction.stock_ticker} at {target_time}.")
                        actual_value = None

            except KeyError: # If target_time is outside available data
                actual_value = None
            
            if actual_value is not None:
                # --- 문제 해결: NumPy float를 Python float로 변환 ---
                # yfinance에서 반환되는 Series의 값은 NumPy float 타입일 수 있습니다.
                # 이를 명시적으로 Python float으로 변환하여 SQLAlchemy 에러를 방지합니다.
                if isinstance(actual_value, (np.float32, np.float64, np.number)): # np.number는 모든 numpy 숫자 타입을 포함
                    actual_value_python = actual_value.item() # .item()으로 Python float으로 변환
                else:
                    actual_value_python = float(actual_value) # 이미 float이거나 다른 숫자 타입일 경우를 대비

                prediction.actual_value = actual_value_python
                
                # error_margin 계산 결과도 NumPy float일 수 있으므로 변환합니다.
                error_margin_calculated = abs(prediction.predicted_value - actual_value_python) / actual_value_python
                if isinstance(error_margin_calculated, (np.float32, np.float64, np.number)):
                    prediction.error_margin = error_margin_calculated.item()
                else:
                    prediction.error_margin = float(error_margin_calculated)

                prediction.is_correct = prediction.error_margin <= tolerance_percent
                db.add(prediction) # Mark as dirty for update
                logger.info(f"Verified prediction {prediction.id} for {prediction.stock_ticker} @ {prediction.predicted_value_for_time}: Predicted={prediction.predicted_value:.4f}, Actual={actual_value_python:.4f}, Error={prediction.error_margin:.2%}, Correct={prediction.is_correct}")
            else:
                logger.warning(f"Failed to find actual value for prediction {prediction.id} (Stock: {prediction.stock_ticker}, Time: {prediction.predicted_value_for_time}). Keeping actual_value as NULL.")

        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error during prediction verification: {e}", exc_info=True)
    finally:
        db.close()

if __name__ == "__main__":
    verify_predictions()