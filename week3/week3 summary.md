## 4.6. XGBoost(eXtra Gradient Boost)
### XGBoost 개요
 - 트리 기반의 앙상블 학습에서 가장 각광받고 있는 알고리즘 중 하나
 - 캐글 경연 대회에서 상위를 차지한 많은 데이터 과학자가 XGBoost를 이용하면서 널리 알려짐
 - XGBoost 주요 장점
 
 |항목|설명|
 |----|---|
 |뛰어난 예측 성능|일반적으로 분류와 회귀 영역에서 뛰어난 예측 성능을 발휘함|
 |GBM 대비 빠른 수행 시간|GBM은 순차적으로 weak learner가 가중치를 증감하는 방법으로 학습이기 때문에 전반적으로 속도가 느림|
 ||하지만 XGBoost는 병렬 수행 및 다양한 기능으로 GBM에 비해 빠른 수행 성능을 보장|
 |과적합 규제|표준 GBM의 경우 과적합 규제 기능이 없으나 XGBoost는 자체에 과적합 규제 기능으로 과적합에 좀 더 강한 내구성을 가질 수 있음|
 |Tree Pruning(나무 가지치기)|일반적으로 GBM은 분할 시 붖어 손실이 발생하면 분할을 더 이상 수행하지 않지만 이러한 방식은 많은 분할을 발생할 수 있음|
 ||XGBoost도 max_depth 파라미터로 분할 깊이를 조정하기도 하지만 나무 가지치기로 더 이상 긍정 이득이 없는 분할을 가지치기 해서 분할 수를 더 줄이는 추가적인 장점을 가지고 있음|
 |자체 내장된 교차 검증|XGBoost는 반복 수행 시마다 내부적으로 학습 데이터 세트와 평가 데이터 세트에 대한 교차 검증을 수행해 최적화된 반복 수행 횟수를 가질 수 있음|
 ||지정된 반복 횟수가 아니라 교차 검증을 통해 평가 데이터 세트의 평가 값이 최적화되면 반복을 중간에 멈출 수 있는 조기 중단 기능이 있음|
 |결손값 자체 처리|XGBoost는 결손값을 자체 처리할 수 있는 기능을 가지고 있음|
 
 - 구분을 짓기 위해 초기의 독자적인 XGBoost 프레임워크 기반의 XGBoost : 파이썬 래퍼 XGBoost 모듈
 - 사이킷런과 연동되는 모듈 : 사이킷런 래퍼 XGBoost 모듈
 
### 파이썬 래퍼 XGBoost 하이퍼 파라미터
  - XGBoost는 GBM과 유사한 하이퍼 파라미터를 동일하게 가지고 있으며 여기에 조기 중단, 과적합을 규제하기 위한 하이퍼 파라미터 등이 추가됐음
  - 파이썬 래퍼 XGBoost 하이퍼 파라미터를 유형별로 나누면 다음과 같음
    - 일반 파라미터 : 일반적으로 실행 시 스레드의 개수나 silent 모드 등의 선택을 위한 파라미터로서 디폴트 파라미터 값을 바꾸는 경우는 거의 없음
      - booster, silent, nthread
    - 부스터 파라미터 : 트리 최적화, 부스팅, regularization 등과 관련 파라미터 등을 지칭합니다.
      - eta, num_boost_rounds, min_child_weight, gamma, max_depth, sub_sample, colsample_bytree, lambda, alpha, scale_pos_weight
    - 학습 태스크 파라미터  : 학습 수행 시의 객체 함수, 평가를 위한 지표 등을 설정하는 파라미터
      - objective, binary, multi:softmax, multi:softprob, eval_metric
  - 뛰어난 알고리즘일수록 파라미터를 튜닝할 필요가 적음
  - 파라미터 튜닝에 들이는 공수 대비 성능 향상 효과가 높지 않은 경우가 대부분
  - 과적합 문제가 심각할 경우 적용할 것들
    - eta 값을 낮춘다 (0.01 ~ 0.1) eta 값을 낮출 경우, num_round(또는 n_estimators)는 반대로 높여줘야 함
    - max_depth 값을 낮춘다.
    - min_child_weight 값을 높인다.
    - gamma 값을 높인다
    - subsample과 colsample_bytree를 조정하는 것도 트리가 너무 복잡하게 생성되는 것을 막음
    
### 사이킷런 래퍼 XGBoost의 개요 및 적용
 - 기존의 XGBoost 모듈에서 사용하던 네이티브 하이퍼 파라미터 몇 개를 변경함
   - eta -> learning_rate
   - sub_sample -> subsample
   - lambda -> reg_lambda
   - alpha -> reg_alpha
 - 조기 중단 파라미터 
   - early_stopping_rounds : 평가 지표가 향상될 수 있는 반복 횟수를 정의
   - eval_metric : 조기 중단을 위한 평가 지표
   - eval_set : 성능 평가를 수행할 데이터 세트
 - 조기 중단 값을 너무 급격하게 줄이면 예측 서능이 저하될 우려가 큼
 

## 4.7. LightGBM
 - XGBoost와 함께 부스팅 계열 알고리즘에서 가장 각광 받고 있음
 - XGBoost가 여전히 학습 시간이 오래 걸림
 - 그에 비해 LightGBM은 XGBoost보다 학습에 걸리는 시간이 훨씬 적음, 또한 메모리 사용량도 상대적으로 적음
 - LightGBM과 XGBoost의 예측 성능은 별다른 차이가 없음
 - LightGBM이 기능상의 다양성이 약간 더 많음
 - 한 가지 단점 : 적은 데이터 세트에 적용할 경우 과적합이 발생하기 쉽다는 것
   - LightGBM의 공식 문서에서 적은 데이터 세트 기준 : 10,000건 이하의 데이터 세트
 - LigthGBM 이론
   - 일반 GBM 계열의 트리 분할 방법과 다르게 리프 중심 트리 분할 방식을 사용
   - 기존의 대부분 트리 기반 알고리즘은 트리 깊이를 효과적으로 줄이기 위한 균형 트리 분할 방식을 사용함
     - 최대한 균형 잡힌 트리를 유지하면서 분할하기 때문에 트리의 깊이가 최소화될 수 있음
     - 균형 잡힌 트리를 생성하는 이유 : 오버피팅에 보다 더 강한 구조를 가질 수 있다고 알려져 있음
     - 그런데 균형을 맞추기 위한 시간이 필요하다는 상대적인 단점이 있음
   - 리프 중심 트리 분할 방식은 트리의 균형을 맞추지 않고 최대 손실 값(max delta loss)을 가지는 리프 노드를 지속적으로 분할하면서 트리의 깊이가
     깊어지고 비대칭적인 규칙 트리가 생성됨
   - 최대 손실값을 가지는 리프 노드를 지속적으로 분할해 생성된 규칙 트리는 학습을 반복할수록 결국은 균형 트리 분할 방식보다 예측 오류 손실을
     최소화 할 수 있다는 것이 LightGBM의 구현 사상
 - LightGBM의 XGBoost 대비 장점
   - 더 빠른 학습과 예측 수행 시간
   - 더 작은 메모리 사용량
   - 카테고리형 피처의 자동 변환과 최적 분할(원-핫 인코딩 등을 사용하지 않고도 카테고리형 피처를 최적으로 변환하고 이에 따른 노드 분할 수행)
   

### LightGBM 하이퍼 파라미터
 - LightGBM 하이퍼 파라미터는 XGBoost와 많은 부분이 유사함
 - 주의해야할 점 : LightGBM은 Xgboost와 다르게 리프 노드가 계속 분할되면서 트리의 깊이가 깊어지므로 이러한 트리 특성에 맞는 하이퍼 파라미터 설정이 필요함(max_depth를 매우 크게 가짐)
 - 주요 파라미터 
   - num_iterations, learning_rate, max_depth, min_data_in_leaf, num_leaves, boosting, bagging_fraction, feature_fraction, lambda_l2, lambda_l1
 
 - Learning Task 파라미터
   - objective

### 하이퍼 파라미터 튜닝 방안
 - 기본 튜닝 방안 : num_learves의 개수를 중심으로 min_child_samples(min_data_in_leaf), max_depth를 함께 조정하면서 모델의 복잡도를 줄이는 것
   - num_leaves 
     - 개별 트리가 가질 수 있는 최대 리프의 개수
     - LightGBM 모델의 복잡도를 제어하는 주요 파라미터
     - num_leaves의 개수를 높이면 정확도가 높아짐
     - 트리의 깊이가 깊어지고 모델이 복잡도가 커져 과적합 영향도가 커짐
   - min_data_in_leaf
     - 사이킷런 래퍼 클래스에서는 min_child_samples로 이름이 바뀜
     - 과적합을 개선하기 위한 중요한 파라미터
     - num_leaves와 학습 데이터의 크기에 따라 달라지지만 보통 큰 값으로 설정하면 트리가 깊어지는 것을 방지
   - max_depth
     - 명시적으로 깊이의 크기를 제한함
     - num_leaves min_data_in_leaf와 결합해 과적합을 개선하는데 사용함
     
 - learning_rate를 작게 하면서 n_estimators를 크게 하는 것은 부스팅 계열 튜닝에서 가장 기본적인 튜닝 방안
   - 물론 n_estimators를 너무 크게 하는 것은 과적합으로 오히려 성능이 저하될 수 있음
 - 과적합을 제어하기 위한 방법
   - reg_lambda, reg_alpha와 같은 regularization을 적용
   - 학습 데이터에 사용할 피처의 개수나 데이터 샘플링 레코드 개수를 줄이기 위해 colsample_bytree, subsample 파라미터를 적용할 수 있음
   
### 파이썬 래퍼 LightGBM과 사이킷런 래퍼 XGBoost, LightGBM 하이퍼 파라미터 비교

|유형|파이썬 래퍼 LightGBM|사이킷런 래퍼 LightGBM|사이킷런 래퍼 XGBoost|
|----|-------------------|---------------------|-------------------|
|파라미터명|num_iterations|n_estimators|n_estimators|
||learning_rate|learning_rate|learning_rate|
||max_depth|max_depth|max_depth|
||min_data_in_leaf|min_child_samples|N/A|
||bagging_fraction|subsample|subsample|
||feature_fraction|colsample_bytree|colsample_bytree|
||lambda_l2|min_child_samples|N/A|
 
