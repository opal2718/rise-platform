import pickle
import numpy as np
import pandas as pd
import shap
import warnings # 경고 메시지 처리를 위한 라이브러리 추가
import os
from google.cloud import storage
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
from constants import no_financial_feature_columns, full_feature_columns, financial_feature_columns_list
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For testing, allow all origins
    allow_methods=["*"], # For testing, allow all methods
    allow_headers=["*"],
)

# --- 1. 가상의 블랙박스 모델 생성 및 저장 (예시를 위한 단계) ---
# 이 부분은 이미 수행되었다고 가정합니다.


GCS_PATH = "/mnt/gcs"

#--- NEW: 두 가지 모델 파일 경로 정의 ---

MODEL_WITH_FINANCIALS_FILENAME = "model_with_financials.pkl"
MODEL_NO_FINANCIALS_FILENAME = "model_no_financials.pkl"
MODEL_WITH_FINANCIALS_FILEPATH = os.path.join(GCS_PATH, MODEL_WITH_FINANCIALS_FILENAME)
MODEL_NO_FINANCIALS_FILEPATH = os.path.join(GCS_PATH, MODEL_NO_FINANCIALS_FILENAME)
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
import sys
# === CRITICAL FIX: Ensure Log1pTransformer is available in the __main__ module for pickling ===
# When the model was pickled, if the training script was run directly,
# Log1pTransformer was in the '__main__' module.
# When uvicorn runs this app, this file becomes '__main__'.
# So, we explicitly ensure the class is set in sys.modules['__main__']
# for pickle to find it correctly during deserialization.
sys.modules['__main__'].Log1pTransformer = Log1pTransformer


with open(MODEL_WITH_FINANCIALS_FILEPATH, "rb") as f:
    with_financial = pickle.load(f)
with open(MODEL_NO_FINANCIALS_FILEPATH, "rb") as f:
    no_financial = pickle.load(f)
# 1-1. 제공된 feature 예시 딕셔너리
sample_input_dict = {
    "PBR": 24.918706757972032, "PER": 32.39435455500224, "ROR": 25.268013788254887, "Close": 0.9200000166893005, "Volume": 65400.0, "Close_lag_1": 0.9200000166893005, "Close_lag_2": 0.9559999704360962, "Close_lag_3": 0.8999999761581421, "Close_lag_4": 0.8889999985694885, "Close_lag_5": 0.8550000190734863, "Close_lag_6": 0.949999988079071, "Close_lag_7": 0.9940000176429749, "Close_lag_8": 0.9800000190734863, "Close_lag_9": 1.0, "Close_lag_10": 1.0299999713897705, "Close_lag_11": 1.0299999713897705, "Close_lag_12": 1.0299999713897705, "Close_lag_13": 1.0299999713897705, "Close_lag_14": 1.0700000524520874, "Close_lag_15": 1.1299999952316284, "Close_lag_16": 1.1200000047683716, "Close_lag_17": 1.090000033378601, "Close_lag_18": 1.0199999809265137, "Close_lag_19": 1.0399999618530273, "Close_lag_20": 1.0700000524520874, "Close_lag_21": 1.0199999809265137, "Close_lag_22": 1.190000057220459, "Close_lag_23": 1.1649999618530273, "Close_lag_24": 1.2200000286102295, "Close_lag_25": 1.440000057220459, "Close_lag_26": 1.3700000047683716, "Close_lag_27": 1.2899999618530273, "Close_lag_28": 1.2400000095367432, "Close_lag_29": 1.2799999713897705, "Close_lag_30": 1.1699999570846558, "Close_lag_31": 1.0800000429153442, "Close_lag_32": 1.1100000143051147, "Close_lag_33": 1.0499999523162842, "Close_lag_34": 1.159999966621399, "Close_lag_35": 0.949999988079071, "Close_lag_36": 1.0099999904632568, "Close_lag_37": 0.9470000267028809, "Close_lag_38": 0.9800000190734863, "Close_lag_39": 0.9739999771118164, "Close_lag_40": 0.9800000190734863, "Close_lag_41": 0.9200000166893005, "Close_lag_42": 0.8500000238418579, "Close_lag_43": 0.8510000109672546, "Close_lag_44": 0.8349999785423279, "Close_lag_45": 0.8199999928474426, "Close_lag_46": 0.7900000214576721, "Close_lag_47": 0.8100000023841858, "Close_lag_48": 0.8100000023841858, "Close_lag_49": 0.8199999928474426, "Close_lag_50": 0.8140000104904175, "Close_lag_51": 0.7699999809265137, "Close_lag_52": 0.7680000066757202, "Close_lag_53": 0.7599999904632568, "Close_lag_54": 0.7749999761581421, "Close_lag_55": 0.7680000066757202, "Close_lag_56": 0.6899999976158142, "Close_lag_57": 0.699999988079071, "Close_lag_58": 0.7200000286102295, "Close_lag_59": 0.7799999713897705, "Close_lag_60": 0.7329999804496765, "Close_lag_61": 0.609000027179718, "Close_lag_62": 0.6290000081062317, "Close_lag_63": 0.6919999718666077, "Close_lag_64": 0.8500000238418579, "Close_lag_65": 0.9580000042915344, "Close_lag_66": 1.0700000524520874, "Close_lag_67": 1.0499999523162842, "Close_lag_68": 1.0399999618530273, "Close_lag_69": 1.0800000429153442, "Close_lag_70": 1.0499999523162842, "Close_lag_71": 0.9900000095367432, "Close_lag_72": 0.9100000262260437, "Close_lag_73": 0.8600000143051147, "Close_lag_74": 0.7879999876022339, "Close_lag_75": 0.7279999852180481, "Close_lag_76": 0.7850000262260437, "Close_lag_77": 77100.0, "Close_lag_78": 0.9279999732971191, "Close_lag_79": 0.8579999804496765, "Close_lag_80": 0.8399999737739563, "Close_lag_81": 0.7770000100135803, "Close_lag_82": 0.7699999809265137, "Close_lag_83": 0.7250000238418579, "Close_lag_84": 0.699999988079071, "Close_lag_85": 0.6480000019073486, "Close_lag_86": 0.6299999952316284, "Close_lag_87": 0.6299999952316284, "Close_lag_88": 0.6690000295639038, "Close_lag_89": 0.7269999980926514, "Close_lag_90": 0.640999972820282, "Volume_lag_1": 65400.0, "Volume_lag_2": 233300.0, "Volume_lag_3": 146700.0, "Volume_lag_4": 185100.0, "Volume_lag_5": 102100.0, "Volume_lag_6": 191200.0, "Volume_lag_7": 144000.0, "Volume_lag_8": 219300.0, "Volume_lag_9": 77100.0, "매출액CFS": -149175128.07660162, "매출액OFS": -404882819.8755175, "자본금CFS": 948844602.8997235, "자본금OFS": 578552713.669075, "Volume_lag_10": 171500.0, "Volume_lag_11": 185700.0, "Volume_lag_12": 155000.0, "Volume_lag_13": 263000.0, "Volume_lag_14": 257500.0, "Volume_lag_15": 388800.0, "Volume_lag_16": 271400.0, "Volume_lag_17": 162300.0, "Volume_lag_18": 245500.0, "Volume_lag_19": 231300.0, "Volume_lag_20": 79600.0, "Volume_lag_21": 613100.0, "Volume_lag_22": 414800.0, "Volume_lag_23": 297200.0, "Volume_lag_24": 885700.0, "Volume_lag_25": 2151600.0, "Volume_lag_26": 1299700.0, "Volume_lag_27": 301800.0, "Volume_lag_28": 794700.0, "Volume_lag_29": 673000.0, "Volume_lag_30": 338000.0, "Volume_lag_31": 143000.0, "Volume_lag_32": 373400.0, "Volume_lag_33": 696800.0, "Volume_lag_34": 2565900.0, "Volume_lag_35": 205700.0, "Volume_lag_36": 229900.0, "Volume_lag_37": 110300.0, "Volume_lag_38": 157700.0, "Volume_lag_39": 140400.0, "Volume_lag_40": 145900.0, "Volume_lag_41": 241800.0, "Volume_lag_42": 160900.0, "Volume_lag_43": 109800.0, "Volume_lag_44": 67300.0, "Volume_lag_45": 146900.0, "Volume_lag_46": 126200.0, "Volume_lag_47": 97300.0, "Volume_lag_48": 80800.0, "Volume_lag_49": 125700.0, "Volume_lag_50": 142800.0, "Volume_lag_51": 104000.0, "Volume_lag_52": 165600.0, "Volume_lag_53": 192100.0, "Volume_lag_54": 392400.0, "Volume_lag_55": 547300.0, "Volume_lag_56": 278600.0, "Volume_lag_57": 210400.0, "Volume_lag_58": 433100.0, "Volume_lag_59": 367100.0, "Volume_lag_60": 555800.0, "Volume_lag_61": 423700.0, "Volume_lag_62": 424000.0, "Volume_lag_63": 355800.0, "Volume_lag_64": 242000.0, "Volume_lag_65": 242700.0, "Volume_lag_66": 177000.0, "Volume_lag_67": 250100.0, "Volume_lag_68": 162700.0, "Volume_lag_69": 285200.0, "Volume_lag_70": 575500.0, "Volume_lag_71": 323000.0, "Volume_lag_72": 258300.0, "Volume_lag_73": 221600.0, "Volume_lag_74": 285100.0, "Volume_lag_75": 339800.0, "Volume_lag_76": 154600.0, "Volume_lag_77": 301700.0, "Volume_lag_78": 248100.0, "Volume_lag_79": 160900.0, "Volume_lag_80": 236200.0, "Volume_lag_81": 325300.0, "Volume_lag_82": 199500.0, "Volume_lag_83": 172200.0, "Volume_lag_84": 222700.0, "Volume_lag_85": 31000.0, "Volume_lag_86": 32700.0, "Volume_lag_87": 133700.0, "Volume_lag_88": 148100.0, "Volume_lag_89": 660200.0, "Volume_lag_90": 49800.0, "부채총계CFS": 219315158.6139207, "부채총계OFS": -812995886.1190152, "영업이익CFS": 241226977.41044998, "영업이익OFS": -628876091.4102951, "유동부채CFS": -393177386.26079035, "유동부채OFS": 682514026.78843, "유동자산CFS": 585069314.8085263, "유동자산OFS": -722071695.7087005, "자본총계CFS": -109262274.86537218, "자본총계OFS": -674194697.2002742, "자산총계CFS": -759851798.5776782, "자산총계OFS": -589994994.330852, "비유동부채CFS": -415523708.01528645, "비유동부채OFS": -96868624.22705507, "비유동자산CFS": -901171955.9122078, "비유동자산OFS": -376319219.40052915, "이익잉여금CFS": 495396229.1159849, "이익잉여금OFS": 103248491.82217145, "총포괄손익CFS": 629723371.0582559, "총포괄손익OFS": -204871215.5443027, "stock_sentiment_avg": 0.3813360623619406, "sector_relevance_avg": 0.08916013310611282, "sector_sentiment_avg": -0.056940482501387446, "stock_news_total_count": 88, "당기순이익(손실)CFS": -589744805.0652394, "당기순이익(손실)OFS": -769929274.2200239, "법인세차감전 순이익CFS": -598442310.8929522, "법인세차감전 순이익OFS": -750112599.4380511
}

# feature_names는 모델 학습 시 사용된 컬럼 순서와 동일하게 유지하는 것이 중요합니다.
feature_names = list(sample_input_dict.keys())
# --- 핵심 수정: predict_function 래퍼 정의 ---
def model_predict_wrapper(input_array, loaded_model):
    """
    shap.KernelExplainer가 NumPy 배열을 전달할 때, 모델의 predict_one이
    DataFrame의 각 행을 딕셔너리로 기대한다면 이를 변환하여 예측합니다.
    """
    input_df = pd.DataFrame(input_array, columns=feature_names)
    predictions = []
    # DataFrame의 각 행을 딕셔너리로 변환하여 predict_one에 전달
    for _, row in input_df.iterrows():
        # 경고를 무시하여 river 내부의 warning 메시지가 SHAP 출력을 방해하지 않도록 합니다.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions.append(loaded_model.predict_one(row.to_dict()))
    return np.array(predictions) # NumPy 배열로 반환

# --- 2. pkl 파일에서 모델 불러오기 및 SHAP 적용 ---
@app.post("/explain")
async def explain(data: dict, request: Request):
    
    stock = data["text"]
    is_korean_stock = stock.endswith(".KS")

    data_url = f"http://34.16.110.5:8001/data?stock={stock}&period=1d" # '1d' 또는 '1h' 등 적절한 period
    
    try:
        response = requests.get(data_url, timeout=30) # 타임아웃 30초 설정
        response.raise_for_status() # HTTP 오류가 발생하면 예외 발생
        data = response.json()
        
        processed_X_for_prediction = data.get("processed_features_for_prediction")
        latest_timestamp_str = data.get("latest_timestamp")
        if processed_X_for_prediction:
            # ai2 API에서 받은 features에 None이 있다면 그대로 유지
            # Sentiment 관련 필드는 0으로 채워질 것이며, 재무 데이터는 없으면 None일 것임
            logger.warning("no warning")
        else:
            logger.warning(f"No 'processed_features_for_prediction' found in AI2 API response for {stock}.")
    except requests.exceptions.Timeout:
        logger.error(f"AI2 API request timed out for {stock}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from AI2 API for {stock}: {e}")

    if not processed_X_for_prediction or not latest_timestamp_str:
        logger.error("Invalid or missing 'processed_features_for_prediction' or 'latest_timestamp' from /data endpoint.")
        raise HTTPException(status_code=422, detail="Invalid processed features format available for prediction.")

    last_timestamp_of_features = pd.to_datetime(latest_timestamp_str)
    prediction_target_time = last_timestamp_of_features + pd.Timedelta(days=1)

    logger.info(f"Processed single instance for model (raw from /data): {processed_X_for_prediction}")
    logger.info(f"Prediction target time based on features: {prediction_target_time}")



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
    loaded_model = with_financial if has_financial_data else no_financial


    instance_to_explain_df = pd.DataFrame([sample_input_dict])
    instance_to_explain_array = instance_to_explain_df.values[0] # NumPy array for explanation
    #print(f"설명할 인스턴스 (유일한 feature 예시):\n{instance_to_explain_df}\n")

    # --- 배경 데이터 (background_data) 생성 ---
    # 주의: 이 부분은 실제 훈련 데이터셋 (또는 대표 샘플)이 여기에 로드되어야 합니다.
    # 현재는 예시를 위해 임의의 값을 가진 DataFrame을 생성합니다.
    # 실제 상황에서는 X_train.sample(n=100, random_state=42) 등으로 대체해야 합니다.
    num_background_samples = 100
    # --- 배경 데이터 (background_data) 생성 ---
    # 실제 훈련 데이터셋의 통계량 (평균, 표준편차 등)을 사용하여
    # 가상의 데이터를 생성하거나, 실제 훈련 데이터에서 샘플링하는 것이 가장 좋습니다.
    # 여기서는 예시를 위해 feature_names의 각 특성에 대해
    # instance_to_explain_df의 값과 유사한 범위의 임의 값을 생성합니다.

    num_background_samples = 100
    synthetic_background_data_df = pd.DataFrame(index=range(num_background_samples), columns=feature_names)

    ratio = 0.5

    # 각 feature_name에 대해 instance_to_explain_df의 값 주변으로 임의의 데이터를 생성
    for col in feature_names:
        if col in instance_to_explain_df.columns:
            # 설명할 인스턴스의 해당 feature 값
            original_value = instance_to_explain_df[col].iloc[0]

            # 숫자형 특성에 대해서만 랜덤 값을 생성
            if pd.api.types.is_numeric_dtype(instance_to_explain_df[col]):
                # original_value가 0이면 0으로 채우고, 아니면 10% 내외의 범위에서 랜덤 값 생성
                if original_value == 0:
                    synthetic_background_data_df[col] = 0.0
                else:
                    # 10% 내외의 범위: original_value * 0.9 ~ original_value * 1.1
                    # 음수 값도 고려하여 min/max를 올바르게 설정
                    lower_bound = original_value * (1-ratio) if original_value > 0 else original_value * (1+ratio)
                    upper_bound = original_value * (1+ratio) if original_value > 0 else original_value * (1-ratio)
                    synthetic_background_data_df[col] = np.random.uniform(min(lower_bound, upper_bound), max(lower_bound, upper_bound), num_background_samples)
            else:
                # 숫자형이 아닌 특성 (예: 범주형)은 원본 값을 그대로 반복
                synthetic_background_data_df[col] = original_value
        else:
            # sample_input_dict에 없는 feature (현재 코드에서는 모든 feature_names가 sample_input_dict에 있음)
            # 이 경우는 발생하지 않을 것이지만, 만약을 위해 임의의 기본값으로 채울 수 있습니다.
            # 예를 들어, 해당 특성의 일반적인 분포에 맞는 임의의 값을 생성합니다.
            synthetic_background_data_df[col] = np.random.rand(num_background_samples) # 0과 1 사이의 임의 값으로 채움

    synthetic_background_data = synthetic_background_data_df.values

    print(f"SHAP 배경 데이터 (가상 생성, {num_background_samples}개 샘플):\n{synthetic_background_data_df.head()}\n")
    explainer = shap.KernelExplainer(
        lambda x: model_predict_wrapper(x, loaded_model), # 래핑 함수 사용
        synthetic_background_data,
        link="identity"
    )

    # instance_to_explain_array가 1D 배열이지만, explainer는 2D 입력을 예상할 수 있습니다.
    # 명시적으로 2D로 만들어서 전달하는 것이 안전합니다.
    shap_values_obj = explainer(instance_to_explain_array.reshape(1, -1))
    shap_values = shap_values_obj.values[0] # 단일 인스턴스이므로 첫 번째 요소를 가져옵니다.
    base_value = shap_values_obj.base_values # 이 부분이 NumPy 배열일 가능성이 있음

    # 2-5. 설명 결과 출력
    print(f"--- SHAP 설명 결과 (딕셔너리 입력, predict_one 사용) ---")
    # 실제 예측도 래퍼 함수와 동일한 방식으로 호출해야 일관성이 있습니다.
    # instance_to_explain_df도 단일 행이므로 to_dict('records')[0]을 사용합니다.
    actual_prediction = loaded_model.predict_one(instance_to_explain_df.iloc[0].to_dict())
    print(f"모델의 실제 예측: {actual_prediction}")

    # base_value가 NumPy 배열일 경우 첫 번째 요소만 사용하거나 item() 메서드 사용
    if isinstance(base_value, np.ndarray):
        print(f"SHAP Base Value (배경 데이터 평균 예측): {base_value.item():.4f}\n") # .item() 사용
    else:
        print(f"SHAP Base Value (배경 데이터 평균 예측): {base_value:.4f}\n")


    print("가장 중요한 Feature 기여도 (SHAP 값):")
    sorted_indices = np.argsort(np.abs(shap_values))[::-1]
    for i in sorted_indices[:10]:
        print(f"- {feature_names[i]}: {shap_values[i]:.40f}")
    return shap_values.tolist()
    # 시각화 코드도 base_value 타입을 고려해야 함 (Jupyter Notebook 환경에서)
    # shap.initjs()
    # if isinstance(base_value, np.ndarray):
    #     html_plot = shap.force_plot(base_value.item(), shap_values, instance_to_explain_array, feature_names=feature_names)
    # else:
    #     html_plot = shap.force_plot(base_value, shap_values, instance_to_explain_array, feature_names=feature_names)
    # shap.save_html('shap_explanation_dict_input.html', html_plot)
    # print("\nSHAP 설명이 완료되었습니다. `shap_explanation_dict_input.html` 파일을 열어 시각화된 결과를 확인하세요.")