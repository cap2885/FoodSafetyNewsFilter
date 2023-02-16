import pickle
import pandas as pd
from konlpy.tag import Mecab
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tqdm import tqdm
from news_scraping import foodnews_scraper



Scraper = foodnews_scraper.news_scraper(strdate, enddate)

