import pandas as pd


INTERNATIONAL_STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL"]
    
"""",AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V",
    "JNJ", "PG", "UNH", "HD", "MA", "DIS", "NFLX", "ADBE", "CRM", "XOM",
    "CVX", "KO", "PEP", "WMT", "NKE", "AMD", "INTC", "CSCO", "CMCSA", "T"
]"""

FULL_DATA_FETCH_RANGE = "1y"
STOCK_PERIOD = "1d"


TRAINING_WINDOW_TIMEDELTA = pd.Timedelta(days=90)
LAG_PERIODS = 90
MODEL_WITH_FINANCIALS_SAVE_PATH = "initializers/model_with_financials.pkl"
MODEL_NO_FINANCIALS_SAVE_PATH = "initializers/model_no_financials.pkl"

base_feature_columns = ['Close', 'Volume', 'stock_sentiment_avg', 'stock_news_total_count', 'sector_sentiment_avg', 'sector_relevance_avg']
lagged_base_features = [f'{col}_lag_{i}' for col in ['Close', 'Volume'] for i in range(1, LAG_PERIODS + 1)]
# DART API Key
DART_API_KEY = "a575bfd53e3345ea7cb70d1eb8106cb03f2de5d7"
# --- Configuration ---
KOREAN_STOCK_TICKERS = [
    "005930.KS", "000660.KS", "035420.KS", "000020.KS", "005380.KS"
]



# NOTE: financial_feature_columns_list는 get_full_financial_report 함수에서
# 'row_name'으로 생성되는 이름들과 일치해야 합니다.
# 예를 들어, '유동자산CFS', '당기순이익(손실)CFS', 'PBR' 등이 정확해야 합니다.
financial_feature_columns_list = [
    '유동자산CFS', '비유동자산CFS', '자산총계CFS', '유동부채CFS', '비유동부채CFS',
    '부채총계CFS', '자본금CFS', '이익잉여금CFS', '자본총계CFS', '매출액CFS',
    '영업이익CFS', '법인세차감전 순이익CFS', '당기순이익(손실)CFS', '총포괄손익CFS',
    '유동자산OFS', '비유동자산OFS', '자산총계OFS', '유동부채OFS', '비유동부채OFS',
    '부채총계OFS', '자본금OFS', '이익잉여금OFS', '자본총계OFS', '매출액OFS',
    '영업이익OFS', '법인세차감전 순이익OFS', '당기순이익(손실)OFS', '총포괄손익OFS',
    'PBR', 'PER', 'ROR'
]
log_features_common = ['Volume'] + [f'Volume_lag_{i}' for i in range(1, LAG_PERIODS + 1)]
log_features_with_financials = financial_feature_columns_list + log_features_common
feature_columns_with_financials = base_feature_columns + lagged_base_features + financial_feature_columns_list
feature_columns_no_financials = base_feature_columns + lagged_base_features

full_feature_columns = base_feature_columns + lagged_base_features + financial_feature_columns_list
no_financial_feature_columns = base_feature_columns + lagged_base_features
