## 4.4. 랜덤포레스트
 - 같은 알고리즘으로 여러 개의 분류기를 써서, 보팅으로 최종 결정
 - 개별 트리가 학습하는 데이터 세트는 전체 데이터세트에서 중첩되게 샘플링된 데이터 세트
 - 중첩된 각각의 데이터 세트에 결정트리 분류기를 각각 적용하는 것이 랜덤 포레스트
 - 부트스트래핑 : 중첩이 되게 sampling 방법
 - 학습 속도가 빠르다
 
### 랜덤포레스트 Hyper Parameter
 - n_estimator : 랜덤포레스트에서 결정트리의 개수(default : 10)
 - max_feature : 트리에서 최적의 분할을 위해 고려할 최대 피쳐 개수	(default : auto)	
 - max_depth/min_sample_leaf
 
## 4.5. GBM(Gradient Boosting Machine)
 - AdaBoost와 유사하나 가중치 업데이트를 경사하강법을 이용
 - 반복 수행을 통해 오류를 최소화 할 수 있도록 가중치를 업데이트 값을 도출하는 기법
 - GBM Hyper Parameter
  - **loss** : 경사 하강법에 사용할 비용함수
  - **learning_rate** : GBM이 학습을 진행할때 마다 적용하는 학습률
  - **n_estimator** : weak learner의 개수
  - **subsample** : weak learner가 학습하는데 사용하는 데이터 샘플링 비율




