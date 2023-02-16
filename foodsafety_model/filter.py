import pickle
import pandas as pd
from konlpy.tag import Mecab
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tqdm import tqdm
from news_scraping import foodnews_scraper

# tokenizer 불러오기(판별모델 생성시의 만든 케라스 토크나이저 모델)
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# mecab 불러오기
tokenizer_mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
# 식품안전뉴스 판별모델 불러오기
loaded_model = load_model('./best_model_26_1119.h5')
# 불용어 리스트
stopwords_df = pd.read_excel('.\stopwords.xlsx')
stopwords = list(stopwords_df['stopwords'])

# 한글자 리스트
onewords_df = pd.read_excel('.\oneword.xlsx')
onewords = list(onewords_df['onewords'])

max_len = 10 # 10으로 패딩

# 식품안전뉴스 분류
def foodnews_predict(new_sentence):
  global score, label, remove_one_word
  new_sentence = re.sub(r'\[[^)]*\]','', new_sentence) # [] 제거
  new_sentence = re.sub(r'[^\uAC00-\uD7A3a-zA-Z\s]','', new_sentence) # 한글 영어 제외 제거
  new_sentence = tokenizer_mecab.morphs(new_sentence) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  
  remove_one_word = [x for x in new_sentence if len(x) > 1 or x in onewords] # 한글자 리스트를 제외한 한글자 제거
  encoded = tokenizer.texts_to_sequences([remove_one_word]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    # print("{:.2f}% 확률로 식품관련 뉴스입니다.\n".format(score * 100))
    score = score*100
    label = "식품관련뉴스O"
  else:
    # print("{:.2f}% 확률로 식품관련 뉴스가 아닙니다.\n".format((1 - score) * 100))
    score = score*100
    label = "식품관련뉴스X"

sbs = pd.read_csv('D:/foodnews_analysis/news_collected_data/notfoodnews_title_230105~0111.csv')
sbs_title = sbs['title']
result_df = pd.DataFrame(columns = ['제목','clean_input','encoded_input', '점수', '예측결과'])
i=0
for news in tqdm(sbs_title):
    foodnews_predict(news)
    result_df.loc[i] = [news, remove_one_word, score, label]
    i+=1

result_df.to_excel(f"./foodnews_predict_result/신경망예측결과_{date}_길이제한x.xlsx", encoding= 'utf-8-sig', index=False)
    
