import requests
import zipfile
import io
import xml.etree.ElementTree as ET
import pandas as pd
import yfinance as yf
import os
import datetime
import json
from google.cloud import storage # GCS 클라이언트 라이브러리 임포트

# --- 설정 (경로 변경 가능) ---
# GCP Cloud Storage 버킷 경로로 변경
# 예: "gs://your-gcs-bucket-name/corp_codes.csv"
CORP_CODES_FILE_PATH = "gs://rise_oml_pickle/corp_codes.csv"

# --- 전역 변수 ---
_CORP_CODES_DF = None
_DART_API_KEY_GLOBAL = None
_GCS_CLIENT = None # GCS 클라이언트 인스턴스를 위한 전역 변수

# --- 내부 유틸리티 함수 ---

def _get_gcs_client():
    """GCS 클라이언트를 초기화하거나 기존 클라이언트를 반환합니다."""
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT

def _load_corp_codes():
    """
    CORP_CODES_FILE_PATH에 지정된 GCS 파일을 로드하고 메모리에 캐시합니다.
    GCS 경로에서 파일을 읽어와 DataFrame으로 만듭니다.
    """
    global _CORP_CODES_DF
    if _CORP_CODES_DF is not None:
        return _CORP_CODES_DF

    if not CORP_CODES_FILE_PATH.startswith("gs://"):
        raise ValueError(f"CORP_CODES_FILE_PATH가 유효한 GCS 경로여야 합니다: {CORP_CODES_FILE_PATH}")

    # GCS 경로 파싱
    path_parts = CORP_CODES_FILE_PATH[5:].split("/", 1)
    if len(path_parts) < 2:
        raise ValueError(f"유효한 GCS 파일 경로가 아닙니다: {CORP_CODES_FILE_PATH}")
    bucket_name = path_parts[0]
    blob_name = path_parts[1]

    print(f"GCS 버킷 '{bucket_name}'에서 파일 '{blob_name}' 로드 시도...")

    try:
        client = _get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # GCS에서 파일 내용을 메모리로 읽어옴
        file_content = blob.download_as_bytes()
        
        # BytesIO를 사용하여 pandas가 메모리에서 파일을 읽도록 함
        _CORP_CODES_DF = pd.read_csv(io.BytesIO(file_content), dtype={"corp_code": str, "stock_code": str})
        print(f"GCS 파일 '{CORP_CODES_FILE_PATH}' 로드 완료. ({len(_CORP_CODES_DF)}개 기업)")
        return _CORP_CODES_DF
    except Exception as e:
        print(f"GCS 파일 '{CORP_CODES_FILE_PATH}' 로드 실패: {e}")
        raise

def _name2code_optimized(corp_name):
    """
    메모리에 캐시된 corp_codes DataFrame에서 기업 이름으로 코드를 검색합니다.
    """
    df_corp_codes = _load_corp_codes()
    if df_corp_codes is None:
        return None

    # corp_code와 stock_code는 str로 읽혔으므로, 검색 시에도 str로 비교하는 것이 안전합니다.
    # 만약 숫자형으로 검색할 경우, df_corp_codes[col].astype(str) == corp_name 과 같이 명시적으로 변환 필요.
    search_columns = ["corp_code", "corp_name", "corp_eng_name", "stock_code"]
    for col in search_columns:
        if col in df_corp_codes.columns:
            # 대소문자 무시하고 부분 일치 검색 (더 유연하게)
            # exact match를 원하면 df_corp_codes[col] == corp_name
            # result = df_corp_codes[df_corp_codes[col].astype(str).str.contains(corp_name, case=False, na=False)]
            # 정확한 일치 검색 (기존 로직 유지)
            result = df_corp_codes[df_corp_codes[col] == corp_name]
            if not result.empty:
                return result.iloc[0]

    print(f"'{corp_name}'을(를) 찾을 수 없습니다.")
    return None

# --- 외부 노출 함수 ---

def initialize_dart_data(api_key: str):
    """
    DART API 키를 설정하고, corp_codes.csv 파일을 로드합니다.
    이 함수는 다른 DART 관련 함수를 호출하기 전에 한 번만 호출되어야 합니다.
    corp_codes.csv 파일은 'CORP_CODES_FILE_PATH' 경로(GCS)에 미리 존재해야 합니다.

    Args:
        api_key (str): 금융감독원 전자공시시스템(DART) API 키.
    """
    global _DART_API_KEY_GLOBAL
    _DART_API_KEY_GLOBAL = api_key
    try:
        _ = _load_corp_codes()
        print("DART 데이터 초기화 완료.")
    except Exception as e:
        print(f"DART 데이터 초기화 실패: {e}")
        raise

def get_full_financial_report(corp_identifier: str, bsns_year: str = None) -> pd.DataFrame | None:
    """
    기업 식별자(이름 또는 코드)를 받아 재무제표와 주요 투자 지표를 반환합니다.
    이 함수를 호출하기 전에 'initialize_dart_data(api_key)'를 먼저 호출해야 합니다.
    이제 'thstrm_amount' (당기 금액)만 포함됩니다.

    Args:
        corp_identifier (str): 기업 이름 또는 corp_code. 예: "삼성전자", "005930"
        bsns_year (str, optional): 사업 연도. 기본값은 현재 연도의 이전 연도입니다.
                                   예: "2023"

    Returns:
        pd.DataFrame: 재무제표와 PBR, PER, ROR이 추가된 DataFrame.
                      조회 실패 시 None 반환.
    """
    if _DART_API_KEY_GLOBAL is None:
        raise ValueError("DART API 키가 설정되지 않았습니다. 'initialize_dart_data'를 먼저 호출하세요.")

    corp_info = _name2code_optimized(corp_identifier)

    if corp_info is None:
        print(f"기업 정보(corp_code, stock_code)를 찾을 수 없습니다: {corp_identifier}")
        return None

    corp_code = corp_info.get("corp_code")
    corp_name = corp_info.get("corp_name", corp_identifier)
    stock_code = corp_info.get("stock_code")

    if not corp_code:
        print(f"'{corp_name}'에 대한 corp_code를 찾을 수 없습니다.")
        return None
    if not stock_code or pd.isna(stock_code):
        print(f"'{corp_name}'에 대한 stock_code를 찾을 수 없습니다.")
        return None

    # 사업 연도 설정 (기본값: 현재 연도의 이전 연도)
    if bsns_year is None:
        bsns_year = str(datetime.datetime.now().year - 1)

    print(f"'{corp_name}' ({corp_code}, {stock_code})의 {bsns_year}년 재무제표 조회...")

    params = {
        "crtfc_key": _DART_API_KEY_GLOBAL,
        "corp_code": corp_code,
        "bsns_year": bsns_year,
        "reprt_code": "11011",  # 11011: 사업보고서
    }

    try:
        res = requests.get("https://opendart.fss.or.kr/api/fnlttSinglAcnt.json", params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
    except requests.exceptions.RequestException as e:
        print(f"DART 재무제표 API 요청 실패: {corp_name} - {e}")
        return None
    except json.JSONDecodeError:
        print(f"DART 재무제표 API 응답 JSON 파싱 실패: {corp_name}")
        return None

    if data.get("status") != "000":
        print(f"{corp_name} 재무제표 조회 실패: {data.get('message', '알 수 없는 오류')}")
        return None

    df_raw = pd.DataFrame(data["list"])
    if df_raw.empty:
        print(f"'{corp_name}'의 {bsns_year}년 재무제표 데이터가 없습니다.")
        return None

    # 'thstrm_amount'만 남기도록 수정
    df_raw['row_name'] = df_raw['account_nm'] + df_raw['fs_div']

    # 'thstrm_amount'만 숫자형으로 변환
    df_raw['thstrm_amount'] = pd.to_numeric(
        df_raw['thstrm_amount'].astype(str).str.replace(',', ''),
        errors='coerce'
    ).fillna(0).astype(int)

    # DataFrame 생성 시 'thstrm_amount'만 선택
    new_df = df_raw[['row_name', 'thstrm_amount']].set_index('row_name')

    # --- 주가 정보 조회 ---
    stock_ticker_yf = f"{stock_code}.KS" if not str(stock_code).endswith(".KS") else str(stock_code)
    stock_value = None
    try:
        stock_data = yf.Ticker(stock_ticker_yf).history(period="1d")
        if stock_data.empty:
            print(f"'{stock_ticker_yf}' 주가 정보를 찾을 수 없습니다. PBR, PER, ROR 계산을 건너뜜니다.")
        else:
            stock_value = stock_data["Close"].iloc[-1]
            print(f"'{stock_ticker_yf}' 현재 종가: {stock_value:,.2f}")
    except Exception as e:
        print(f"'{stock_ticker_yf}' 주가 정보 조회 실패: {e}. PBR, PER, ROR 계산을 건너뜜니다.")

    # --- PBR, PER, ROR 계산 ---
    # 새 DataFrame은 'thstrm_amount' 컬럼만 있으므로, 여기에 추가합니다.
    # 인덱스가 'PBR', 'PER', 'ROR'인 행을 추가하고 'thstrm_amount'에 값을 넣습니다.
    # loc를 사용할 때 존재하지 않는 인덱스와 컬럼을 동시에 지정하면 자동으로 생성됩니다.
    new_df.loc['PBR', 'thstrm_amount'] = pd.NA
    new_df.loc['PER', 'thstrm_amount'] = pd.NA
    new_df.loc['ROR', 'thstrm_amount'] = pd.NA

    if stock_value is not None:
        # CF_S를 포함한 계정명이 정확히 일치해야 합니다. (예: '자본총계CFS')
        total_capital = new_df.loc['자본총계CFS', 'thstrm_amount'] if '자본총계CFS' in new_df.index else pd.NA
        net_income = new_df.loc['당기순이익(손실)CFS', 'thstrm_amount'] if '당기순이익(손실)CFS' in new_df.index else pd.NA
        total_assets = new_df.loc['자산총계CFS', 'thstrm_amount'] if '자산총계CFS' in new_df.index else pd.NA

        # PBR, PER, ROR 계산 시 'thstrm_amount' 컬럼에만 값을 할당
        if pd.notna(total_capital) and total_capital != 0:
            new_df.loc['PBR', 'thstrm_amount'] = stock_value / total_capital

        if pd.notna(net_income) and net_income != 0:
            new_df.loc['PER', 'thstrm_amount'] = stock_value / net_income

        if pd.notna(net_income) and pd.notna(total_assets) and total_assets != 0:
            new_df.loc['ROR', 'thstrm_amount'] = net_income / total_assets

    return new_df