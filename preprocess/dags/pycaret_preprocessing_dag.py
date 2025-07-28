
from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
import pandas as pd
import os

def preprocess_for_pycaret():
    """
    PyCaret 결과의 main.py 스크립트 로직을 기반으로 데이터를 전처리합니다.
    train과 test 데이터를 읽고, 시간 기반 특성을 생성하며,
    제출용 테스트 세트를 준비합니다.
    """
    # 현재 스크립트의 디렉토리를 기준으로 상대 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..")  # preprocess/dags에서 프로젝트 루트로
    
    # 상대 경로 정의
    data_path = os.path.join(project_root, "DATA")
    output_path = os.path.join(project_root, "preprocess", "processed_data")
    os.makedirs(output_path, exist_ok=True)

    # 데이터 로드
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    # '일시'를 datetime 객체로 변환
    train_df['date_time'] = pd.to_datetime(train_df['일시'])
    test_df['date_time'] = pd.to_datetime(test_df['일시'])

    # 테스트 세트에 대해 'num_date_time' 생성 (제출용)
    # 형식: {건물번호}_{YYYYMMDDHH}
    test_df['num_date_time'] = test_df['건물번호'].astype(str) + '_' + test_df['date_time'].dt.strftime('%Y%m%d%H')

    # 원본 스크립트와 같이 시간 특성 추출
    for df in [train_df, test_df]:
        df['month'] = df['date_time'].dt.month
        df['day'] = df['date_time'].dt.day
        df['time'] = df['date_time'].dt.hour
        df.drop(columns=['date_time', '일시'], inplace=True)

    # 전처리된 데이터 저장
    train_df.to_csv(os.path.join(output_path, "train_pycaret_preprocessed.csv"), index=False)
    # 제출 형식에 맞게 num_date_time을 인덱스로 설정
    test_df.set_index('num_date_time', inplace=True)
    test_df.to_csv(os.path.join(output_path, "test_pycaret_preprocessed.csv"), index=True)


with DAG(
    dag_id="pycaret_preprocessing",
    catchup=False,
    schedule=None,
    tags=["dacon", "electricity", "pycaret"],
    doc_md="pycaret_result/main.py 스크립트의 로직을 기반으로 데이터를 전처리하는 DAG입니다."
) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_for_pycaret",
        python_callable=preprocess_for_pycaret,
    )
