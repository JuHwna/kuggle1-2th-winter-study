#
## 1. 사이킷런의 설명

![scikitlearn method](https://user-images.githubusercontent.com/49123169/72659123-3c0a3600-39fe-11ea-9f33-d2ea5241fcfa.PNG)

## 2. 첫 번째 머신러닝 만들어보기
### 1) 붓꽃 데이터(Iris data)로 품종 예측하기

## 3. 사이킷런의 기반 프레임워크 익히기
### 1) Estimator 이해 및 fit(), predict() 메서드

![estimator](https://user-images.githubusercontent.com/49123169/72659282-7d034a00-3a00-11ea-8d3b-11ed7ee14ff8.PNG)

- 사이킷런의 주요 모듈

|분류|모듈명|설명|
|----------------|--------------------|-------------------------------------------------------------------------------------------|
|예제 데이터|sklearn.datasets|사이킷런에 내장되어 예제로 제공하는 데이터 세트|
|피처 처리|sklearn.preprocessing|데이터 전처리에 필요한 다양한 가공 기능 제공(문자열을 숫자형 코드 값으로 인코딩, 정규화, 스케일링 등)|
|피처 처리|sklearn.feature_selection|알고리즘에 큰 영향을 미치는 피처를 우선순위대로 셀렉션 작업을 수행하는 다양한 기능 제공|
|피처 처리|sklearn.feature_extraction|텍스트 데이터나 이미지 데이터의 벡터화된 피처를 추출하는데 사용됨|
|피처 처리 & 차원 축소|sklearn.decomposition|차원 축소와 관련한 알고리즘을 지원하는 모듈|
|테이터 분리, 검증 & 파라미터 튜닝|sklearn.model_selection|교차 검증을 위한 학습용/테스트용 분리, 그리드 서치로 최적 파라미터 추출 등의 API 제공|
|평가|sklearn.metrics|분류, 회귀, 클러스터링, 페어와이즈에 대한 다양한 성능 측정 방법 제공(Accuracy,Precision,Recall,ROC-AUC,RMSE 등)|
|ML 알고리즘|sklearn.ensemble|앙상블 알고리즘 제공(랜덤 포레스트,에이다 부스트, 그래디언트 부스팅 등을 제공)|
|ML 알고리즘|sklearn.linear_model|주로 선형 회귀, 릿지, 라쏘 및 로지스틱 회귀 등 회귀 관련 알고리즘을 지원 또한 SGD 관련 알고리즘도 제공|
|ML 알고리즘|sklearn.naive_bayes|나이브 베이즈 알고리즘 제공, 가우시간 NB, 다항분포 NB 등|
|ML 알고리즘|sklearn.neighbors|최근접 이웃 알고리즘 제공, K-NN 등|
|ML 알고리즘|sklearn.svm|서포트 벡터 머신 알고리즘 제공|
|ML 알고리즘|sklearn.tree|의사 결정 트리 알고리즘 제공|
|ML 알고리즘|sklearn.cluster|비지도 클러스터링 알고리즘 제공(K-평균, 계층형,DBSCAN 등|
|유틸리티|sklearn.pipeline|피처 처리 등의 변환과 ML 알고리즘 학습, 예측 등을 함께 묶어서 실행할 수 있는 유틸리티 제공|

