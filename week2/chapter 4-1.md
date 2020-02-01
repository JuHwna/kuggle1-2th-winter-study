# 4. 분류
## 4.1. 분류의 개요
 - 지도학습 : 답이 존재하는 상태에서 알고리즘을 통해 학습시키고 새로운 데이터 값에 미지의 레이블을 예측하는 것.
 - 앙상블 : 정형데이터의 예측 분석 영역에서 좋은 성능
 
## 4.2.결정트리
 - 앙상블의 기본 알고리즘의 결정트리
 - 데이터의 정규화나 스케일링에 의한 영향이 적음.
 - 예측성능을 높이기 위해 복잡한 규칙을 만들다 보면 과적합 발생
 - 하지만 여러 개의 약한학습기를 결합해 확률적 보완과 오류가 발생한 부분에서 가중치를 계속 업데이트에 보완시키는 앙상블에 있어서 결정트리는 좋은 약한학습기.
 - **결정트리**
   - 데이터에 있는 규칙을 자동으로 찾아내 트리 기반의 분류 규칙을 만드는 것.
   - 깊이(depth) 깊어질수록 과적합확률 
   - 그렇기에 최대한 많은 데이터세트를 분류할 수 있는 규칙노드를 만들어야 함.
   - 또한 분류된 데이터세트들은 최대한 균일한 정보를 가진 데이터들이어야 함.
   ![image](https://user-images.githubusercontent.com/49123169/73587322-e3f32980-44fd-11ea-8def-2f5b073deb2d.png)
   ![image](https://user-images.githubusercontent.com/49123169/73587332-f1101880-44fd-11ea-956b-9a62da0e2830.png)
   
   - 장점 
      - 쉽다, 직관적이다
      - 피처의 스케일링이나 정규화등 사전 가공 영향도가 크지 않다.
   - 단점 
      - 과적합으로 알고리즘 성능이 떨어진다. 이를 극복하기 위해 사전에 트리의 크기를 제한하는 튜닝 필요.
   - 결정트리 파라미터
      - **min_samples_split** : 분할되기 위해 노드가 가져야하는 최소샘플 수
      - **min_samples_leaf** : 리프 노드가 가지고있어야 하는 최소 샘플 수
      - **min_weight_fraction_leaf**: min_samples_leaf와 비슷하지만 가중치가 부여된 전체 샘플 수에서의 비율
      - **max_leaf_nodes**: 리프 노드의 최대수 
      - **max_features**: 각 노드에서 분할에 사용할 특성의 최대 수
~~~
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
df_clf=DecisionTreeClassifier(random_state=156)
df_clf.fit(x_train,y_train)
pred=df_clf.predict(x_test)
accuracy=accracy_score(y_test,pred)
~~~

## Feature_importance
 - 결정트리는 균일도를 기준으로 어떠한 속성을 규칙으로 만드냐가 중요함
 - 몇 개의 중요한 피처를 통해 좀 더 명확한 규칙을 생성할 수 있고 간결한 모형과 이상값에 강한 모형을 만들 수 있음

~~~
import seaborn as sns
importances_values=best_df_clf.feature_importances_
importances=pd.Series(importances_values,index=X_train,columns)
top20=importances.sort_values(ascending=False)
sns.barplot(x=top20,y=top20.index)
~~~
