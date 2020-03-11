# Chapter 8. 텍스트 분석
### NLP이냐 텍스트 분석이냐?
- NLP(National Language Processing) : 머신이 인간의 언어를 이해하고 해석하는데 더 중점을 두고 기술이 발전해 왔음
  - NLP의 영역에는 언어를 해석하기 위한 기계 번역, 자동으로 질문을 해석하고 답을 해주는 질의응답 시스템 등의 영역 등에서 텍스트 분석과 차별점이 있음
  - NLP는 텍스트 분석을 향상하게 하는 기반 기술
  - NLP 기술이 발전함에 따라 텍스트 분석도 더욱 정교하게 발전할 수 있음
    - 예전의 텍스트를 구성하는 언어적인 룰이나 업무의 룰에 따라 텍스트를 분석하는 룰 기반 시스템에서 머신러닝의 텍스트 데이터를 기반으로 모델을 학습하고 
      예측하는 기반으로 변경되면서 많은 기술적 발전이 가능해짐
    
- 텍스트 분석(텍스트 마이닝(Tex Mining)) : 비정형 텍스트에서 의미 이쓴 정보를 추출하는 것에 좀 더 중점을 두고 기술이 발전해 왔음
  - 머신러닝, 언어 이해, 통계 등을 활용해 모델을 수립하고 정보를 추출해 비즈니스 인텔리전스나 예측 분석 등의 분석 작업을 주로 수행
  - 머신러닝 기술에 입어 텍스트 분석은 크게 발전하고 있음
  - 다음과 같은 기술 영역에 집중해왔음
  
  |기술 영역|내용|
  |--------|----|
  |텍스트 분류|Text Categorization이라고도 함. 문서가 특정 분류 또는 카테고리에 속하는 것을 예측하는 기법을 통칭함. 지도학습을 적용함|
  |감성 분석|텍스트에서 나타나는 감정/판단/믿음/의견/기분 등의 주관적인 요소를 분석하는 기법. 텍스트 분석에서 가장 활발하게 사용되고 있는 분야. 지도학습 방법뿐만 아니라 비지도학습을 이용해 적용할 수 있음|
  |텍스트 요약|텍스트 내에서 중요한 주제나 중심 사상을 추출하는 기법. 대표적으로 토픽 모델링이 있음|
  |텍스트 군집화와 유사도 측정|비슷한 유형의 문서에 대해 군집화를 수행하는 기법. 텍스트 분류를 비지도학습으로 수행하는 방법의 일활으로 사용될 수 있음. 유사도 측정 역시 문서들간의 유사도를 측정해 비슷한 문서끼리 모을 수 있는 방법|
  
## 1. 텍스트 분석 이해
- 텍스트 분석 : 비정형 데이터인 텍스트를 분석하는 것
  - 지금까지 ML 모델은 주어진 정형 데이터 기반에서 모델을 수립하고 예측을 수행했음
  - 머신러닝 알고리즘은 숫자형의 피처 기반 데이터만 입력받을 수 있기 때문에 텍스트를 머신러닝에 적용하기 위해서는 비정형 텍스트 데이터를 어떻게 
    피처 형태로 추출하고 추출된 피처에 의미 있는 값을 부여하는가 하는 것이 매우 중요한 요소
- 피처 벡터화(피처 추출): 텍스트를 word(또는 word의 일부분)기반의 다수의 피처로 추출하고 이 피처에 단어 빈도수와 같은 숫자 값을 부여하면 
  텍스트는 단어의 조합인 벡터값으로 표현
  - 대표적으로 텍스트를 피처 벡터화해서 변환하는 방법 : BOW(Bag of Words)와 Word2Vec
- 텍스트를 벡터값을 가지는 피처로 변환하는 것은 머신러닝 모델을 적용하기 전에 수행해야 할 매우 중요한 요소

### 텍스트 분석 수행 프로세스
1. **텍스트 사전 준비작업(텍스트 전처리)** : 텍스트를 피처로 만들기 전에 사전에 클렌징, 대/소문자 변경, 늩ㄱ수문자 삭제 등의 클렌징 작업, 단어 등의 토큰화 작업, 의미 없는 단어(Stop word) 제거 작업, 어근 추출(Stemming/Lemmatization) 등의 텍스트 정규화 작업을 수행하는 것을 통칭
2. **피처 벡터화 추출** : 사전 준비 작업으로 가공된 텍스트에서 피처를 추출하고 여기에 벡터 값을 할당함
   - 대표적인 방법 : BOW와 Word2Vec
     - BOW는 대표적으로 Count 기반과 TF-IDF 기반 벡터화
3. **ML 모델 수립 및 학습/예측/평가** : 피처 벡터화된 데이터 세트에 ML 모델을 적용해 학습/예측 및 평가를 수행함

### 파이썬 기반의 NLP, 텍스트 분석 패키지
- 파이썬 기반에서 NLP와 텍스트 분석을 위해 쉽고 편하게 텍스트 사전 정제 작업, 피처 벡터화/추출, ML모델을 지원하는 매우 훌륭한 라이브러리가 많음(대부분 영어 기반의 라이브러리)
  - NLTK(National Language Toolkit for Python)
    - 파이썬의 가장 대표적인 NLP 패키지
    - 방대한 데이터 세트와 서브 모듈을 가지고 있으며 NLP의 거의 모든 영역을 커버하고 있음
    - 많은 NLP 패키지가 NLTK의 영향을 받아 작성되고 있음
    - 수행 성능과 정확도, 신기술, 엔터프라이즈한 기능 지원 등의 측면세어 부족한 부분이 있음
    - 실제 대량의 데이터 기반에서는 제대로 활용되지 못하고 있음
  - Gensim
    - 토픽 모델링 분야에서 가장 두각을 나타내는 패키지
    - 오래전부터 토픽 모델링을 쉽게 구현할 수 있는 기능을 제공해 왔으며 Word2Vec 구현 등의 다양한 신기능도 제공함
    - SpaCy와 함께 가장 많이 사용되는 NLP 패키지
  - SpaCy 
    - 뛰어난 수행 성능으로 최근 가장 주목을 받는 NLP 패키지
    - 많은 NLP 애플리케이션에서 SpaCy를 사용하는 사례가 늘고 있음
    
- 사이킷런은 머신러닝 위주의 라이브러리여서 NLP를 위한 다양한 라이브러리, 예를 들어 '어근 처리'와 같은 NLP 패키지에 특화된 라이브러리는 가지고 있지 않음
  - 하지만 텍스트를 일정 수준으로 가공하고 머신러닝 알고리즘에 텍스트 데이터를 피처로 처리하기 위한 편리한 기능을 제공하고 있어 사이킷런으로도
    충분히 테스트 분석 기능을 수행할 수 있음
  - 하지만 더 다양한 텍스트 분석이 적용돼야하는 경우, 보통은 NLTK/Gensim/SpaCy와 같은 NLP 전용 패키지와 함께 결합해 애플리케이션을 작성하는 경우 많음
  
## 2. 텍스트 사전 준비 작업(텍스트 전처리) - 텍스트 정규화
- 텍스트 자체를 바로 피처로 만들 수는 없음
  - 이를 위해 사전에 텍스트를 가공하는 준비 작업이 필요함
- 텍스 정규화 : 텍스트를 머신러닝 알고리즘이나 NLP 애플리케이션에 입력 데이터로 사용하기 위해 클렌징, 정제, 토큰화, 어근화 등의 다양한 텍스트 데이터의 사전 작업을 수행하는 것을 의미함
- 텍스트 분석은 이러한 텍스트 작업이 매우 중요함
- 텍스트 정규화 작업
  - 클렌징, 토큰화, 필터링/스톱 워드 제거/철자 수정, Stemming, Lemmatization
  
### 클렌징
- 텍스트에서 분석에 오히려 방해가 되는 불필요한 문자, 기호 등을 사전에 제거하는 작업
- HTML, XML 태그나 특정 기호 등을 사전에 제거

### 텍스트 토큰화
- 토큰화의 유형은 문서에서 문장을 분리하는 문장 토큰화와 문장에서 단어를 토큰으로 분리하는 단어 토큰화로 나눌 수 있음

#### 문장 토큰화
- 문장 토큰화 : 문장의 마침표(.), 개행문자(\n) 등 문장의 마지막을 뜻하는 기호에 따라 분리하는 것
  - 정규 표현식에 따른 문장 토큰화도 가능
#### 단어 토큰화
- 단어 토큰화 : 무낭을 단어로 토큰화하는 것
  - 기본적으로 공백, 콤마(,), 마침표(.), 개행문자 등으로 단어를 분리하지만 정규 표현식을 이용해 다양한 유형으로 토큰화를 수행할 수 있음
- 마침표(.)나 개행문자와 같이 문장을 분리하는 구분자를 이용해 단어를 토큰화할 수 있으므로 Bag of Word와 같이 단어의 순서가 중요하지 않은 경우 문장 토큰화를 사용하지 않고 단어 토큰화만 사용해도 충분함
  - 일반적으로 문장 토큰화는 각 문장이 가지는 시맨틱적인 의미가 중요한 요소로 사용될 때 사용함
  
- 문장을 단어별로 하나씩 토큰화 할 경우 문맥적인 의미는 무시될 수 밖에 없음
  - 이러한 문제를 조금이라도 해결해 보고자 도입된 것이 n-gram
- n-gram : 연속된 n개의 단어를 하나의 토큰화 단위로 분리해 내는 것
  - n개 단어 크기 윈도우를 만들어 문장의 처음부터 오른쪽으로 움직이면서 토큰화를 수행합니다.


### 스톱 워드 제거
- 스톱 워드 : 분석에 큰 의미가 없는 단어를 지칭
  - 가령 영어에서 is, the, a, will 등 문장을 구성하는 필수 문법 요소지만 문맥적으로 큰 의미가 없는 단어가 이에 해당함
  - 이 단어의 경우 문법적인 특성으로 인해 특히 빈번하게 텍스트에 나타나므로 이것들을 사전에 제거하지 않으면 그 빈번함으로 인해 오히려 중요한 단어로 인지될 수 있음
  - 이 의미 없는 단어를 제거하는 것이 중요한 전처리 작업
- 언어별로 이러한 스톱 워드가 목록화돼 있음
  - NLTK의 경우 가장 다양한 언어의 스톱 워드를 제공함
  - NLTK의 스톱 워드에는 어떤 것이 있는지?
    - 먼저 NLTK의 stopwords 목록을 내려받는다
    - NLTK의 경우 단어 사전과 같이 참조가 필요한 데이터 세트의 경우 인터넷으로 내려받게 돼 있음
    - 일단 내려받기가 완료된 경우에는 다시 내려 받지 않지만 최초 내려받기가 필요하기 때문에 수행하려는 컴퓨터에 인터넷 연결이 돼 있는지 먼저 확인해야함
    
### Stemming과 Lemmatization
- 문법적 또는 의미적으로 변화하는 단어의 원형을 찾는 것
  - 두 기능 모두 원형 단어를 찾는다는 목적은 유사하지만 Lemmatization이 stemming보다 정교하며 의미론적인 기반에서 단어의 원형을 찾음
- Stemming : 원형 단어로 변환 시 일반적인 방법을 적용하거나 더 단순화된 방법을 적용해 원래 단어에서 일부 철자가 훼손된 어근 단어를 추출하는 경향
- Lemmatization : 품사와 같은 문법적인 요소와 더 의미적인 부분을 감안해 정확한 철자로 된 어근 단어를 찾아줌
  - Lemmatization이 Stemming보다 변환에 더 오랜 시간을 필요로 함
- NLTK는 다양한 Stemmer를 제공함
  - Porter, Lancaster, Snowball Stemmer가 있음
- Lemmatization은 WordNetLemmatizer를 제공함

## 3. Bag of Words - BOW
- Bag of Words 모델 : 문서가 가지는 모든 단어를 문맥이나 순서를 무시하고 일괄적으로 단어에 대해 빈도 값을 부여해 피처 값을 추출하는 모델
- Bag of Words(BOW) 모델 : 문서 내 모든 단어를 한꺼번에 봉투 안에 넣은 뒤에 흔들어서 섞는다는 의미
- BOW 모델의 장점
  - 쉽고 빠른 구축
  - 단순히 단어의 발생 횟수에 기반하고 있지만 예상보다 문서의 특징을 잘 나타낼 수 있는 모델이어서 전통적으로 여러 분야에서 활용도가 높음
- BOW 모델의 단점
  - 문맥 의미 반영 부족 : BOW는 단어의 순서를 고려하지 않기 때문에 문장 내에서 단어의 문맥적인 의미가 무시됨. 
    - 물론 이를 보완하기 위해 n_gram 기법을 활용할 수 있지만 제한적인 부분에 그치므로 언어의 많은 부분을 차지하는 문맥적인 해석을 처리하지 못하는 단점
  - 희소 행렬 문제(희소성, 희소 행렬) : BOW로 피처 벡터화를 수행하면 희소 행렬 형태의 데이터 세트가 만들어지기 쉬움. 
    - 많은 문서에서 단어를 추출하면 매우 많은 단어가 칼럼으로 만들어짐. 
    - 문서마다 서로 다른 단어로 구성되기에 단어가 문서마다 나타나지 않는 경우가 훨씬 더 많음. 
    - 즉 매우 많은 문서에서 단어의 총 개수는 수만 ~ 수십만 개가 될 수 있는데 하나의 문서에 있는 단어는 이 중 극히 일부분이므로 대부분의 데이터는 0값으로 채워지게 됨. 
    - 희소 행렬 : 대규모의 칼럼으로 구성된 행렬에서 대부분의 값이 0으로 채워지는 행렬
    - 밀집 행렬 : 대부분의 값이 0이 아닌 의미 있는 값으로 채워져 있는 행렬
    - 희소 행렬 : 일반적으로 ML 알고리즘의 수행 시간과 예측 성능을 떨어뜨리기 때문에 희소 행렬을 위한 특별한 기법이 마련돼 있음

### BOW 피처 벡터화
- 피처 벡터화 : 텍스트의 특정 의미를 숫자형 값인 벡터 값으로 변환하는 것
  - 각 문서의 텍스트를 단어로 추출해 피처로 할당하고 각 단어의 발생 빈도와 같은 값을 이 피처에 값으로 부여해 각 문서를 이 단어 피처의 발생 빈도 값으로 구성된 벡터로 만드는 기법
  - 기존 텍스트 데이터를 또 다른 형태의 피처의 조합으로 변경하기 때문에 넓은 범위의 피처 추출에 포함됨
  - 텍스트 분석에서는 피처 벡터화와 피처 추출을 같은 의미로 사용하곤 함
- BOW 모델에서 피처 벡터화를 수행한다는 것 : 모든 문서에서 모든 단얼르 칼럼 형태로 나열하고 각 문서에서 해당 단어의 횟수나 정규화된 빈도를 값으로 부여하는 데이터 세트 모델로 변경하는 것

- BOW의 피처 벡터화의 방식
  - 카운트 기반의 벡터화
    - 단어 피처에 값을 부여할 때 각 문서에서 해당 단어가 나타나는 횟수, 즉 Count를 부여하는 경우임
    - 카운트 값이 높을수록 중요한 단어로 인식됨
    - 그러나 카운트만 부여할 경우 그 문서의 특징을 나타내기보다는 언어의 특성상 문장에서 자주 사용될 수 밖에 없는 단어까지 높은 값을 부여하게 됨
  - TF-IDF 벡터화
    - 카운트 기반의 벡터화의 문제 보완
    - 개별 문서에서 자주 나타나는 단어에 높은 가중치를 주되, 모든 문서에서 전반적으로 자주 나타나는 단어에 대해서는 페널티를 주는 방식으로 값을 부여함
    - 문서마다 텍스트가 길고 문서의 개수가 많은 경우 카운트 방식보다는 TF-IDF 방식을 사용하는 것이 더 좋은 예측 성능을 보장할 수 있음
    
### 사이킷런의 Count 및 TF-IDF 벡터화 구현 : CountVectorizer, TfidVectorizer
- 카운트 기반의 벡터화를 구현한 클래스 : 사이킷런의 CountVectorizer 클래스
  - 단지 피처 벡터화만 수행하지는 않으며 소문자 일괄 변환, 토큰화, 스톱 워드 필터링 등의 텍스트 전처리도 함께 수행함
  - 텍스트 전처리 및 피처 벡터화를 위한 입력 파라미터를 설정해 동장함
  - fit()과 transform()을 통해 피처 벡터화된 객체를 변환함
  - CountVectorizer의 입력 파라미터
  
    