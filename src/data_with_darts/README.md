# Darts 활용하기

### 목표

> Darts를 이용해 데이터 전처리 파이프라인을 구축하고, 자체 예측 모델을 학습 및 평가하는 파이프라인 구축하기




### 사용 모델

1단계: LightGBMModel(퀀타일: 0.1/0.5/0.9)로 빠른 베이스라인

2단계: TFTModel로 다변량+past/future 공변량 반영, 확률예측

3단계: NHiTS/NBEATS 추가 후 RegressionEnsembleModel로 앙상블


### Darts 구현 팁

#### TFTModel

- past_covariates: 과거 기상 실측, 과거 수요 파생

- future_covariates: 캘린더, 기상 예보

- static_covariates: 설비 특성, 구역/고객군 더미

#### LightGBMModel

- feature_dim = 랙+롤링+더미+예보 피처; quantile_loss로 불확실성 추정

#### 앙상블

- RegressionEnsembleModel로 P50 기준 성능 극대화

