import datetime
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import requests
from foodnews_scraper import news_scraper
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time
from selenium.common import exceptions

# 14개의 언론사 수집
print("수집기간을 설정하세요")    
strdate_entry = input('시작날짜를 입력하세요 "YYYY-MM-DD" -> ')
stryear, strmonth, strday = map(int, strdate_entry.split('-'))
strdate = datetime.date(stryear, strmonth, strday)
enddate_entry = input('마지막날짜를 입력하세요 "YYYY-MM-DD" -> ')
endyear, endmonth, endday = map(int, enddate_entry.split('-'))
enddate = datetime.date(endyear, endmonth, endday)

print(strdate, enddate)
Scraper = news_scraper(strdate, enddate)

# foodjournal_df = Scraper.foodjournal()
# foodjournal_df.to_csv('foodjournal.csv', encoding = 'utf-8-sig', index=False)
# foodbeverage_df = Scraper.foodbeverage()
# foodbeverage_df.to_csv('foodbeverage.csv', encoding = 'utf-8-sig', index=False)
# foodtoday_df = Scraper.foodtoday()
# foodtoday_df.to_csv('foodtoday.csv', encoding = 'utf-8-sig', index=False)
# naverfood_df = Scraper.naverfood()
# naverfood_df.to_csv('naverfood.csv', encoding = 'utf-8-sig', index=False)

# df_list = [foodjournal_df, foodbeverage_df, foodtoday_df, naverfood_df]

# df_news = pd.concat(df_list, ignore_index=True)
# df_news = df_news.sort_values(by=['posted_date', 'source'])
# df_news.to_csv('foodnews_title.csv', encoding = 'utf-8-sig', index=False)

# # 식품관련x뉴스
# notnaverfood_df = Scraper.notnaverfood_title()
# notnaverfood_df.to_csv('D:/foodnews_analysis/news_collected_data/notfoodnews_title_230110.csv', encoding= 'utf-8-sig', index=False)

