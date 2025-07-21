import pandas as pd
import numpy as np
import random
import os
# PyCaret 라이브러리 추가 (GPU 지원)
from pycaret.regression import setup, compare_models, predict_model, finalize_model, create_model, tune_model, evaluate_model

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
    return df


def main(num_building):
    train_df = preprocess_data('./open/train.csv', num_building)
    test_df = preprocess_data('./open/test.csv', num_building)

    train_x = train_df.drop(columns=['일사(MJ/m2)', '일조(hr)', '전력소비량(kWh)'])
    train_y = train_df['전력소비량(kWh)']

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
        ignore_features=['num_date_time', '일시', '건물번호', '일조(hr)', '일사(MJ/m2)'],
        train_size=0.8,  # 학습 데이터 비율 증가 (GPU 효율성)
        fold_strategy='kfold',
        fold=5,
        # GPU 최적화를 위한 추가 설정
        numeric_imputation='mean',
        categorical_imputation='mode',
        transformation=True,
        normalize=True,
        pca=False,
        # GPU 메모리 최적화
        # low_variance_threshold=0.1,
        # multicollinearity_threshold=0.95
    )

    best_models = []
    print("✅ PyCaret 환경 설정 완료 - GPU 가속 모드 활성화")

    best_models = compare_models(
        include=['lightgbm', 'rf', 'et', 'xgboost', 'catboost'],
        sort='MAPE',
        n_select=5,
        verbose=True,
        # parallel=True,
        budget_time=20
    )
    
    best_model = best_models[0]

    # GPU 최적화된 튜닝 설정
    tuned_model = tune_model(
    best_model, 
    optimize='MAPE', 
    n_iter=10,  # 반복 횟수 증가 (GPU로 빠르게 처리)
    search_library='optuna',  # 고급 최적화 라이브러리
    search_algorithm='tpe',  # Tree-structured Parzen Estimator
    early_stopping=True,  # 조기 종료
    early_stopping_max_iters=10,
    verbose=True
    )

    # 바로 예측해보기
    predictions = predict_model(tuned_model, data=test_df)

    # 헤더 없이, append 모드로 저장 (겹쳐지게)
    csv_name = f'./output/pycaret_tuned_submission.csv'

    if os.path.exists(csv_name):
        predictions.to_csv(csv_name, index=True, mode='a', header=False)
    else:
        print(f'{csv_name} 파일이 없습니다. 새로 생성합니다.')
        header = ['num_date_time', 'answer']
        predictions.to_csv(csv_name, index=True, header=header)


if __name__ == "__main__":
    # for i in range(1, 101):
    #     main(i)
    main(1)