from get_data import get_processed_features_for_stock
import pandas as pd
pd.set_option('display.max_columns', None)  # 모든 컬럼을 표시
pd.set_option('display.max_rows', None)     # 모든 행을 표시 (여기서는 1행이므로 큰 의미는 없음)
#pd.set_option('display.width', 1000)        # 출력 너비 설정 (컬럼이 많을 때 줄바꿈 방지)
print(get_processed_features_for_stock("AAPL", "2024-06-30", "1d", 90, False, True))#, "2024-06-30"))