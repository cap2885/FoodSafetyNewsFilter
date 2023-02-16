import pandas as pd
import re
from konlpy.tag import Mecab
import collections
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM, SimpleRNN
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import gensim
from gensim.models import KeyedVectors
from tensorflow.keras.initializers import Constant
import joblib
from datetime import datetime
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as stats

# mecab tokenizer 
tokenizer_mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

news_path = "D:/foodnews_analysis/news_collected_data/"
# 식품관련 뉴스 불러오기
df_food_3year = pd.read_csv(r'D:\foodnews_analysis\find_foodnews\sim_safetynews2.csv') # 식품관련뉴스 불러오기
# 식품관련 뉴스 라벨링
df_food = df_food_3year.copy()
df_food['label'] = 1
print("식품관련 뉴스 총 개수 :", len(df_food))
print(df_food.head())
# 식품뉴스 불용어 리스트
stopwords_df = pd.read_excel(r'D:/foodnews_analysis/find_foodnews\food_stopwords.xlsx')
stopwords = list(stopwords_df['stopwords'])
# 한글자 리스트
onewords_df = pd.read_excel('D:/foodnews_analysis/find_foodnews\oneword.xlsx')
onewords = list(onewords_df['onewords'])
def clean_food(text):
    pattern = re.compile(r'\[[^)]*\]') # []없애기
    pattern2 = re.compile(r'[^\uAC00-\uD7A3a-zA-Z\s]') # 한글 영어만

    result = pattern.sub('',text)
    result = pattern2.sub('', result)
    result = result.replace(u'\xa0', u' ')
    temp = tokenizer_mecab.morphs(result) # 형태소로 나눈 리스트
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    remove_one_word = [x for x in temp if len(x) > 1 or x in onewords] # 한글자 제거 (but, 의미있는것은 놔둠)
    
    return remove_one_word
df_food['title_clean2'] = df_food.title_clean.map(clean_food)
print("식품관련뉴스 데이터 : ", df_food.head())
# 식품관련 X 뉴스 불러오기
sbs1 = pd.read_csv(r'D:\foodnews_analysis\find_foodnews\sbs~40000_32149개.csv')
sbs2 = pd.read_csv(r'D:\foodnews_analysis\find_foodnews\sbs40000~_32858개.csv')
allsbs_df = pd.concat([sbs1,sbs2])

# 식품관련 X 뉴스 합치기
df_notfood = allsbs_df.copy()
print("식품관련 X 뉴스 총 개수 :", len(df_notfood))
print(df_notfood.head())
df_notfood['label'] = 0


# 식품관련 X 뉴스중에서 식품안전 관련 단어가 포함된 제목을 삭제 
foodsafe_word = pd.read_excel('D:/foodnews_analysis/find_foodnews/foodword.xlsx') # 식품안전관련 단어
print(foodsafe_word.head())
sametitle = []
for line in df_notfood['title']:
    if any(word in line for word in list(foodsafe_word['word'])):
        sametitle.append(line)
sametitle_df = pd.DataFrame({'title' : sametitle})
# sametitle_df.to_csv('sametitle.csv', encoding= 'utf-8-sig', index=False)
mask = df_notfood['title'].isin(sametitle)
drop_notfood = df_notfood[~mask]
print("식품안전단어를 포함한 뉴스 삭제후 개수 : ", len(drop_notfood))
# 불용어 리스트
stopwords_df = pd.read_excel('D:/foodnews_analysis/find_foodnews\stopwords.xlsx')
stopwords = list(stopwords_df['stopwords'])
# 한글자 리스트
onewords_df = pd.read_excel('D:/foodnews_analysis/find_foodnews\oneword.xlsx')
onewords = list(onewords_df['onewords'])
# 머신러닝
def clean2(text):
    pattern = re.compile(r'\[[^)]*\]') # []없애기
    pattern2 = re.compile(r'[^\uAC00-\uD7A3a-zA-Z\s]') # 한글 영어만

    result = pattern.sub('',text)
    result = pattern2.sub('', result)
    temp = tokenizer_mecab.morphs(result) # 형태소로 나눈 리스트
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    remove_one_word = [x for x in temp if len(x) > 1 or x in onewords] # 한글자 제거 (but, 의미있는것은 놔둠)
    # encoded_sentence = []
    # for word in result:
    #     try:
    #         # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
    #         encoded_sentence.append(word_to_index[word])
    #     except KeyError:
    #         # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
    #         encoded_sentence.append(word_to_index['OOV'])
    # encoded_sentence = np.array(encoded_sentence)
    return str(remove_one_word) # 신경망은 문자열 X


# 제목 전처리 - 딥러닝
def clean(text):
    pattern = re.compile(r'\[[^)]*\]') # []없애기
    pattern2 = re.compile(r'[^\uAC00-\uD7A3a-zA-Z\s]') # 한글 영어만

    result = pattern.sub('',text)
    result = pattern2.sub('', result)
    result = result.replace(u'\xa0', u' ')
    temp = tokenizer_mecab.morphs(result) # 형태소로 나눈 리스트
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    remove_one_word = [x for x in temp if len(x) > 1 or x in onewords] # 한글자 제거 (but, 의미있는것은 놔둠)
    # encoded_sentence = []
    # for word in result:
    #     try:
    #         # 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴.
    #         encoded_sentence.append(word_to_index[word])
    #     except KeyError:
    #         # 만약 단어 집합에 없는 단어라면 'OOV'의 정수를 리턴.
    #         encoded_sentence.append(word_to_index['OOV'])
    # encoded_sentence = np.array(encoded_sentence)
    return remove_one_word # 신경망은 문자열 X

drop_notfood['title_clean2'] = drop_notfood.title_clean.map(clean)

# 식품관련뉴스 + 식품관련X 뉴스 합치기
df3_list = [df_food, drop_notfood]
allabel_df = pd.concat(df3_list, ignore_index=True)
print("총 뉴스 개수 :", len(allabel_df))
allabel_df['title_clean_str'] = allabel_df.title_clean.map(clean2)
allabel_df.drop_duplicates(subset=['title_clean_str'], inplace=True)
print("중복 제거 후 총 뉴스 개수:", len(allabel_df))
titlelabel_df = allabel_df[['title_clean2','label']]
titlelabel_df.columns = ['title','label']
print("최종 준비 데이터프레임: ")
print(titlelabel_df)
titlelabel_df['count'] = [len(x) for x in titlelabel_df['title']]
print(titlelabel_df.head())
stats.probplot(titlelabel_df['count'], plot=plt)
plt.show()
titlelabel_df2 = titlelabel_df.copy()
# titlelabel_df2['title_clean_list'] = titlelabel_df2.title.map(clean) # 전처리 함수 돌리기
# titlelabel_df2['title_clean_str'] = titlelabel_df2.title.map(clean2) # 전처리 함수 돌리기

# dtmvector = CountVectorizer()
# X_train_dtm = dtmvector.fit_transform(titlelabel_df2['title'])
# x_train, x_test, y_train, y_test = train_test_split(X_train_dtm, titlelabel_df2['label'], test_size=0.2, random_state=777, stratify=titlelabel_df2['label'])
now = datetime.now()
date = str(now.month)+str(now.day)+"_"+str(now.hour)+str(now.minute)
# def make_models(xtrain, xtest, ytrain, ytest):
#     # 로지스틱회귀분석
#     model1 = LogisticRegression(random_state=42)
#     model1.fit(xtrain, ytrain)
#     print('logistic_model :', model1.score(xtest, ytest))
#     joblib.dump(model1, f'./logistic{date}')
#     # SVM
#     model2 = svm.SVC(probability=True)
#     model2.fit(xtrain, ytrain)
#     print('svm_model :', model2.score(xtest, ytest))
#     joblib.dump(model2, f'./svm{date}')
#     # 랜덤포레스트
#     model3 = RandomForestClassifier(random_state=42)
#     model3.fit(xtrain, ytrain)
#     print('rf_model :', model3.score(xtest, ytest))
#     joblib.dump(model3, f'./rf{date}')
#     return model1, model2, model3

# # logistic, svm_model, rf_model = make_models(x_train, x_test, y_train, y_test)

# sbs = pd.read_csv('D:/foodnews_analysis/news_collected_data/notfoodnews_title_230105~0111.csv')
# sbs['clean'] = sbs.title.map(clean2)
# sbs_title = sbs['clean']
# dtm = dtmvector.transform(sbs_title)

# def model_result(model):
#     score = model.predict_proba(dtm)
#     score = np.array(score).T[1]
#     res = model.predict(dtm)
#     res = res.reshape(1,-1)[0]
#     result_df = pd.DataFrame({'제목' : sbs['title'],
#                             '전처리' : sbs_title,
#                             '점수' : score,
#                             '결과' : res})

#     result_df.to_excel(f"./foodnews_predict_result/{model}예측결과_tfidf_{date}.xlsx", encoding= 'utf-8-sig', index=False)

# # model_result(logistic)
# # model_result(svm_model)
# # model_result(rf_model)

preprocessed_sentences = list(titlelabel_df2['title'])
print(preprocessed_sentences[:10])
tokenizer = Tokenizer() # 케라스 토크나이저 모델
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences) # 제목들을 정수인코딩
max_len = max(len(item) for item in encoded) # 최대 길이의 제목
padded = pad_sequences(encoded, padding='post') # 최대길이를 기준으로 패딩

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# vocabsize 지정하기
# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1 #어휘 사전의 크기(빈도수 2이하인 단어는 제거)
print('단어 집합의 크기 :',vocab_size)

# vocab_size 지정 후 정수 인코딩 재진행
tokenizer = Tokenizer(vocab_size) # 케라스 토크나이저 모델
tokenizer.fit_on_texts(preprocessed_sentences)
word_index = tokenizer.word_index # 빈도수가 큰순서대로 dictionary 생성 
encoded = tokenizer.texts_to_sequences(preprocessed_sentences) # 정수 인코딩
print(word_index)

num_tokens = [len(tokens) for tokens in encoded]
print("제목 최대길이 : ", max(len(tokens) for tokens in encoded))
#최대 길이를 (평균 + 2*표준편차)로 계산
print("제목 평균길이 : ",np.mean(num_tokens))
max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens) # 평균 + 2*표준편차
min_tokens = np.mean(num_tokens) - 2*np.std(num_tokens) 
maxlen = int(max_tokens)
print('패딩할 최대 길이(평균 + 2*표준편차) : ', maxlen)
print('전체 문장의 {}%가 {}개 ~ {}개 사이에 포함됨. '.format((np.sum((min_tokens <num_tokens) & (num_tokens < max_tokens)) / len(num_tokens) *100), int(min_tokens), int(max_tokens)))

print('제목의 최대 길이 :',max(len(review) for review in encoded))
print('제목의 평균 길이 :',sum(map(len, encoded))/len(encoded))
plt.hist([len(review) for review in encoded], bins=25)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()


titlelen = []
for t in encoded :
    titlelen.append(len(t))
    
encoded_df = pd.DataFrame({"encoded_title" : encoded,
                           "label" : titlelabel_df2['label'],
                           "titlelen" : titlelen})
print(encoded_df.head())
print("개수",len(encoded_df))
# encoded_df = encoded_df[(10 >= encoded_df["titlelen"]) & (encoded_df['titlelen'] >= 3)]
encoded_remove_short = encoded_df['encoded_title']
print(len(encoded_df))
print('제목의 최대 길이 :',max(len(review) for review in encoded_remove_short))
print('제목의 평균 길이 :',sum(map(len, encoded_remove_short))/len(encoded_remove_short))
plt.hist([len(review) for review in encoded_remove_short], bins=25)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()        

# 제목 길이 비율 구하기
def below_threshold_len(min_len, max_len, nested_list):
  count = 0
  for sentence in nested_list:
    if(min_len <= len(sentence) <= max_len):
        count = count + 1
  print('전체 샘플 중 길이가 %s 이상 %s 이하인 샘플의 비율: %s'%(min_len, max_len, (count / len(nested_list))*100))
  
max_len = 10 # 제목 길이 최대값
min_len = 3 # 제목 길이 최소값
below_threshold_len(min_len, max_len, encoded_remove_short)

padded = pad_sequences(encoded_remove_short)
print(padded[:5])
x_data = padded
y_data = np.array(encoded_df['label'])
# # 최종전처리후학습 데이터 확인
# print(type(x_data), type(y_data))
# result_df = pd.DataFrame({"x_data" : list(x_data),
#                           "y_data" : y_data})
# result_df.to_excel("최종전처리후학습data.xlsx", encoding= 'utf-8-sig', index=False)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=777, stratify=y_data)
print(x_train[:5])


embedding_dim = 32  #워드 벡터의 차원수
hidden_units = 32
        
# model = Sequential() # 계층적 모델 구성
# model.add(Embedding(input_dim = vocab_size, output_dim = embedding_dim)) # 워드 임베딩 레이어 생성
# model.add(SimpleRNN(hidden_units)) # RNN or LSTM모델 사용
# model.add(Dense(1, activation='sigmoid')) # 이진분류이므로 출력층에 활성화함수 sigmoid 사용

# print(model.summary())
# # val_loss가 2번연속 증가하면 학습중지
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
# # val_acc가 좋아질 경우에만 모델 저장
# mc = ModelCheckpoint(f'best_model_{date}.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(x_train, y_train, epochs=4, callbacks=[es, mc], batch_size=64, validation_split=0.2)

# # 훈련 과정 시각화 (정확도)
# plt.plot(history.history['acc'])

# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# # 훈련 과정 시각화 (손실)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Train', 'Val'], loc='upper left')
# plt.show()

# loaded_model = load_model(f'best_model_{date}.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))

# with open('tokenizer_practice.pickle', 'wb') as handle:
#      pickle.dump(tokenizer, handle)
