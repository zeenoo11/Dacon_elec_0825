import pandas as pd
import numpy as np
import random
import os
# PyCaret 라이브러리 추가 (GPU 지원)
from pycaret.regression import setup, compare_models, predict_model, finalize_model, create_model, tune_model, evaluate_model, blend_models

# pycaret 로그 레벨 설정 (디버깅을 위해 INFO로 변경)
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "INFO"

import warnings
warnings.filterwarnings(action='ignore')

# GPU 사용을 위한 환경 설정
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "INFO"  # GPU 상태 확인을 위해 로그 레벨 증가

# GPU 가용성 확인
import torch
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 장치: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
else:
    print("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.") 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

def preprocess_data(file_path, num_building):
    df = pd.read_csv(file_path)
    # 'num_date_time'을 인덱스로 설정
    df = df.set_index('num_date_time')
    df = df[df['건물번호'] == num_building]
    df = df.drop(columns=['건물번호'])
    # '일시'에서 월, 일, 시간 추출
    df['month'] = df['일시'].apply(lambda x: int(x[4:6]))
    df['day'] = df['일시'].apply(lambda x: int(x[6:8]))
    df['time'] = df['일시'].apply(lambda x: int(x[9:11]))
    df = df.drop(columns=['일시'])
    if 'test' in file_path:
        df['일조(hr)'] = 0.0
        df['일사(MJ/m2)'] = 0.0
    return df


def main(num_building):
    train_df = preprocess_data('./open/train.csv', num_building)
    test_df = preprocess_data('./open/test.csv', num_building)

    # PyCaret 환경 설정 (GPU 최적화)
    print("GPU 가속을 위한 PyCaret 환경 설정 중...")

    exp = setup(
        data=train_df,
        target='전력소비량(kWh)',
        session_id=42,
        verbose=False,
        use_gpu=True,  # GPU 사용 활성화
        feature_selection=False,
        remove_multicollinearity=False,
        ignore_features=['num_date_time'],
        train_size=0.8,  # 학습 데이터 비율 증가 (GPU 효율성)
        fold_strategy='kfold',
        fold=5,
        # GPU 최적화를 위한 추가 설정
        numeric_imputation='knn',
        categorical_imputation='mode',
        transformation=True,
        normalize=True,
        pca=False,
        # GPU 메모리 최적화
        # low_variance_threshold=0.1,
        # multicollinearity_threshold=0.95
    )

    print("✅ PyCaret 환경 설정 완료 - GPU 가속 모드 활성화")

    # 상위 2개 모델 비교 (GPU 가속 모델 위주)
    print("상위 모델 비교/선택 중...")
    top_models = compare_models(
        include=['xgboost', 'catboost', 'lightgbm', 'rf', 'et', 'gbr'], # GPU 지원 및 성능이 좋은 모델
        sort='MAPE',
        n_select=3, # 상위 2개 모델 선택
        verbose=True
    )

    # 모델 블렌딩
    print("선택된 모델 블렌딩 중...")
    blended_model = blend_models(
        estimator_list=top_models,
        optimize='MAPE',
        fold=5, # 교차 검증 폴드 수
        verbose=True
    )
    
    # 블렌딩된 모델 최종 학습
    print("블렌딩 모델 최종 학습(Finalizing) 중...")
    final_model = finalize_model(blended_model)

    # 예측 수행
    predictions = predict_model(final_model, data=test_df)
    result = predictions.copy()[["prediction_label"]]
    result.columns = ['answer']

    # 헤더 없이, append 모드로 저장 (겹쳐지게)
    csv_name = f'./output/blended_submission.csv'

    if os.path.exists(csv_name):
        result.to_csv(csv_name, index=True, mode='a', header=False)
    else:
        print(f'{csv_name} 파일이 없습니다. 새로 생성합니다.')
        header = ['answer']
        result.to_csv(csv_name, index=True, header=header)


if __name__ == "__main__":
    for i in range(1, 101):
        main(i)
    # main(1)