# Chapter 6. 차원 축소
## 1. 차원 축소 개요
 - 대표적인 차원 축소 알고리즘 : PCA,LDA,SVD,NMF
 - 차원 축소 : 매우 많은 피처로 구성된 다차원 데이터 세트의 차원을 축소해 새로운 차원의 데이터 세트를 생성하는 것
 - 차원이 증가할수록 데이터 포인트 간의 거리가 기하급수적으로 멀어지게 되고 희소한 구조를 가지게 됨
 - 수백 개 이상의 피처로 구성된 데이터 세트의 경우 상대적으로 적은 차원에서 학습된 모델보다 예측 신뢰도가 떨어짐
 - 피처가 많은 경우 개별 피처 간에 상관관계가 높을 가능성이 큼
   - 선형 회귀와 같은 선형 모델에서는 입력 변수 간의 상관관계가 높을 경우 이로 인한 다중 공선성 문제로 모델의 예측 성능이 저하됨
 - 매우 많은 다차원의 피처를 차원 축소해 피처 수를 줄이면 더 직관적으로 데이터를 해석할 수 있음
 - 차원 축소를 할 경우 학습 데이터의 크기가 줄어들어서 학습에 필요한 처리 능력도 줄일 수 있음
 - 차원 축소는 피처 선택과 피처 추출로 나눌 수 있음
   - 피처 선택(특성 선택) : 말 그대로 특정 피처에 종속성이 강한 부릴요한 피처는 아예 제거하고 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 것
   - 피처 추출 : 기존 피처를 저차원의 중요 피처로 압축해서 추출하는 것
     - 새롭게 추출된 중요 특성은 기존의 피처가 압축된 것이므로 기존의 피처와는 완전히 다른 값이 됨
     - 기존 피처를 단순 압축이 아닌 피처를 함축적으로 더 잘 설명할 수 있는 또 다른 공간으로 매핑해 추출하는 것
     - 함축적인 특성 추출 : 기존 피처가 전혀 인지하기 어려웠던 잠재적인 요소(Latent Factor)를 추출하는 것
 - 차원 축소
   - 단순히 데이터의 압축을 의미하는 것 X
   - 더 중요한 의미 : 차원 축소를 통해 좀 더 데이터를 잘 설명할 수 있는 잠재적인 요소를 추출하는 데 있음
   - PCA, SVD, NMF : 잠재적인 요소를 찾는 대표적인 차원 축소 알고리즘
     - 매우 많은 차원을 가지고 있는 이미지나 텍스트에서 차원 축소를 통해 잠재적인 의미를 찾아 주는 데 이 알고리즘이 잘 활용되고 있음
 - 차원 축소 알고리즘은 매우 많은 피셀로 이뤄진 이미지 데이터에서 잠재된 특성을 피처로 도출해 함축적 형태의 이미지 변환과 압축을 수행할 수 있음
   - 이렇게 변환된 이미지는 원본 이미지보다 훨씬 적은 차원이기 때문에 이미지 분류 등의 분류 수행 시에 과적합 영향력이 작아져서 오히려 
     원본 데이터로 예측하는 것보다 예측 성능을 더 끌어 올릴 수 있음
     
     (그런가? 일단 이미지 수가 적어서 생기는 문제에서 쓰는 것 같은데 이미지 변조를 쓰면 해결 될 문제인 것 같긴 한데 잘 모르겠다)
     
   - 이미지 자체가 가지고 있는 차원의 수가 너무 크기 때문에 비슷한 이미지라도 적은 픽셀의 차이가 잘못된 예측으로 이어질 수 있기 때문
 - 차원 축소 알고리즘이 자주 사용되는 또 다른 영역 : 텍스트 문서의 숨겨진 의미를 추출하는 것
   - 문서 내 단어들의 구성에서 숨겨져 있는 시맨틱 의미나 토픽을 잠재 요소로 간주하고 이를 찾아낼 수 있음
   - SVD와 NMF는 이러한 시맨틱 토픽 모델링을 위한 기반 알고리즘으로 사용됨
   
## 2. PCA(Principal Component Analysis)
### PCA 개요
 - PCA : 가장 대표적인 차원 축소 기법
   - 여러 변수 간에 존재하는 상관관계를 이용해 이를 대표하는 주성분을 추출해 차원을 축소하는 기법
   - 기존 데이터의 정보 유실이 최소화하는 것이 당연
   - 가장 높은 분산을 가지는 데이터의 축을 찾아 이 축으로 차원을 축소하는데 이것이 PCA의 주성분이 됨(즉, 분산이 데이터의 특성을 가장 잘 나타내는 것으로 간주)
   - 데이터 변동성이 가장 큰 방향으로 축을 생성하고 새롭게 생성된 축으로 데이터를 투영하는 방식
 - PCA 진행 방식
   - 제일 먼저 가장 큰 데이터 변동성을 기반으로 첫 번째 벡터 축을 생성
   - 두 번째 축은 이 벡터 축에 직각이 되는 벡터(직교 벡터)를 축으로 함
   - 세 번째 축은 다시 두 번째 축과 직각이 되는 벡터를 설정하는 방식으로 축을 생성
   - 이렇게 생성된 벡터 축에 원본 데이터를 투영하면 벡터 축의 개수만큼의 차원으로 원본 데이터가 차원 축소됨
 - PCA는 원본 데이터의 피처 개수에 비해 매우 작은 주성분으로 원본 데이터의 총 변동성을 대부분 설명할 수 있는 분석법
 - 선형대수 관점에서 해석해 보면 입력 데이터의 공분산 행렬을 고유값 분해하고 이렇게 구한 고유벡터에 입력 데이터를 선형 변환하는 것
   - 이 고유벡터가 PCA의 주성분 벡터로서 입력 데이터의 분산이 큰 방향을 나타냄
   - 고유값은 바로 이 고유벡터의 크기를 나타내며 동시에 입력 데이터의 분산을 나타냄
   
     