import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np # numpy import 추가
import logging
import datetime


# 재무제표 관련 함수 임포트
from rise_proto import initialize_dart_data, get_full_financial_report 

DART_API_KEY = "a575bfd53e3345ea7cb70d1eb8106cb03f2de5d7"
#DART_API_KEY = os.getenv("DART_API_KEY", "YOUR_DART_API_KEY_HERE") 

# DART 데이터 초기화 (애플리케이션 시작 시 한 번만 호출)
dart_initialized_successfully = False
if DART_API_KEY == "YOUR_DART_API_KEY_HERE" or not DART_API_KEY:
    print("WARNING: DART API Key is not set or is default. Financial report features will be disabled.")
else:
    try:
        initialize_dart_data(DART_API_KEY)
        print("DART API 데이터 초기화 성공.")
        dart_initialized_successfully = True
    except Exception as e:
        print(f"ERROR: DART API 데이터 초기화 실패: {e}. Financial report features will be disabled.")
        dart_initialized_successfully = False

LAG_PERIODS = 90
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

feature_columns = ['Close', 'Volume', 'stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']
for i in range(1, LAG_PERIODS + 1):
    feature_columns.append(f'Close_lag_{i}')
    feature_columns.append(f'Volume_lag_{i}')

# NEW: 재무제표 관련 컬럼 추가
# 제공해주신 출력 형태를 참고하여 모든 항목을 추가합니다.
financial_feature_columns = [
    '유동자산CFS', '비유동자산CFS', '자산총계CFS', '유동부채CFS', '비유동부채CFS',
    '부채총계CFS', '자본금CFS', '이익잉여금CFS', '자본총계CFS', '매출액CFS',
    '영업이익CFS', '법인세차감전 순이익CFS', '당기순이익(손실)CFS', '총포괄손익CFS',
    '유동자산OFS', '비유동자산OFS', '자산총계OFS', '유동부채OFS', '비유동부채OFS',
    '부채총계OFS', '자본금OFS', '이익잉여금OFS', '자본총계OFS', '매출액OFS',
    '영업이익OFS', '법인세차감전 순이익OFS', '당기순이익(손실)OFS', '총포괄손익OFS',
    'PBR', 'PER', 'ROR'
]
feature_columns.extend(financial_feature_columns)


# --- REVISED /data endpoint (MODIFIED) ---
@app.get("/data")
async def get_data(stock: str, period: str):
    logger.info(f"Received /data request for stock: {stock}, period: {period}")

    if not stock:
        logger.error("Stock ticker is empty in /data request.")
        raise HTTPException(status_code=400, detail="Stock ticker cannot be empty.")

    required_days = LAG_PERIODS + 30
    yf_data_range = f"{required_days}d"
    logger.info(f"Automatically determined yfinance data range: {yf_data_range} for {stock}")

    df_initial = pd.DataFrame()
    try:
        ticker = yf.Ticker(stock)
        hist = ticker.history(period=yf_data_range, interval=period)

        if hist.empty:
            logger.warning(f"No historical data found for stock {stock} with range {yf_data_range} and period {period}.")
        else:
            df_initial = hist[['Close', 'Volume']].copy()
            logger.info(f"Fetched {len(df_initial)} trend data points for {stock}.")

    except Exception as e:
        logger.error(f"Error fetching historical data for stock {stock} using yfinance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch historical data for {stock}: {e}")

    if df_initial.empty:
        raise HTTPException(status_code=422, detail="No trend data available to process.")

    lagged_features = []
    for i in range(1, LAG_PERIODS + 1):
        lagged_features.append(df_initial['Close'].shift(i).rename(f'Close_lag_{i}'))
        lagged_features.append(df_initial['Volume'].shift(i).rename(f'Volume_lag_{i}'))

    df_trend = pd.concat([df_initial] + lagged_features, axis=1)

    df_trend["stock_sentiment_avg"] = 0.0
    df_trend["stock_news_total_count"] = 0.0
    df_trend["sector_sentiment_avg"] = 0.0
    df_trend["sector_relevance_avg"] = 0.0

    df_trend = df_trend.dropna()

    logger.info(f"Shape of df_trend after lagged features and dropna: {df_trend.shape}")

    if df_trend.empty:
        raise HTTPException(status_code=422, detail=f"Not enough historical data for stock {stock} to generate {LAG_PERIODS} lagged features.")

    # --- NEW: DART 재무제표 정보 가져오기 및 병합 (NaN 처리) ---
    corp_identifier_for_dart = stock.split('.')[0] if '.' in stock else stock
    logger.info(f"Attempting to fetch DART financial report for corp_identifier: {corp_identifier_for_dart}")

    financial_report_df = None
    if dart_initialized_successfully:
        try:
            financial_report_df = get_full_financial_report(corp_identifier_for_dart)
        except Exception as e:
            logger.error(f"Error fetching DART financial report for {corp_identifier_for_dart}: {e}", exc_info=True)
            financial_report_df = None
    else:
        logger.warning("DART API was not initialized successfully. Skipping financial report fetching.")

    last_index = df_trend.index[-1]

    # financial_feature_columns의 모든 항목을 초기화합니다 (기본값 NaN).
    # 이후 financial_report_df에서 찾은 값으로 덮어씁니다.
    for financial_col in financial_feature_columns:
        df_trend.loc[last_index, financial_col] = np.nan # 일단 모든 재무 항목을 NaN으로 초기화

    if financial_report_df is not None and not financial_report_df.empty:
        logger.info(f"Successfully fetched financial report for {corp_identifier_for_dart}. Merging data.")
        financial_data_dict = financial_report_df['thstrm_amount'].to_dict()

        for financial_col in financial_feature_columns:
            # PBR, PER, ROR은 이미 계산되어 financial_report_df에 직접 들어가 있습니다.
            if financial_col in ['PBR', 'PER', 'ROR']:
                val = financial_report_df.loc[financial_col, 'thstrm_amount'] if financial_col in financial_report_df.index else np.nan
            else: # 그 외의 재무 항목은 financial_data_dict에서 가져옵니다.
                val = financial_data_dict.get(financial_col, np.nan)
            
            # DataFrame에 값 할당 (NaN으로 초기화된 값을 덮어씁니다)
            df_trend.loc[last_index, financial_col] = val
            logger.debug(f"Added financial feature {financial_col}: {df_trend.loc[last_index, financial_col]}")
    else:
        logger.warning(f"No financial report available or DART API failed for {corp_identifier_for_dart}. Financial features remain NaN.")
        # 이 경우, 위에서 np.nan으로 초기화했으므로 추가 작업 불필요.


    # --- 가장 마지막 행만 추출하여 딕셔너리로 반환 ---
    latest_features_row = df_trend.iloc[-1]
    latest_timestamp = df_trend.index[-1].isoformat()

    processed_features_for_prediction = {}
    for col in feature_columns:
        value = latest_features_row.get(col)
        # np.nan을 JSON 호환성을 위해 None으로 변환
        if pd.isna(value):
            processed_features_for_prediction[col] = None 
        else:
            processed_features_for_prediction[col] = float(value)

    logger.info(f"Sending single processed feature instance for prediction with NaN/None handling for financial features.")
    logger.info(f"Latest timestamp of features: {latest_timestamp}")
    
    return {
        "processed_features_for_prediction": processed_features_for_prediction,
        "latest_timestamp": latest_timestamp
    }