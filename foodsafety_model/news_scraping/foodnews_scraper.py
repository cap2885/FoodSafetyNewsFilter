# 패키지 불러오기 
from datetime import datetime
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
import requests
import time
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time
from selenium import webdriver
import pandas as pd


class news_scraper:
    def __init__(self, strdate, enddate):
        self.start = strdate
        self.end = enddate

    def foodjournal(self): # 식품저널뉴스 수집기함수

        # make dataframe
        foodjournal_df = pd.DataFrame(columns=("posted_date","posted_date(time)","title","source"))
        idx=0
        pagenum = 1
        source = '식품저널뉴스'
        # exploration from 1 page
        while True:
            print(pagenum)

            # call 1page url
            listurl = "https://www.foodnews.co.kr/news/articleList.html?page=" + str(pagenum) + "&total=69060&box_idxno=&sc_section_code=S1N1&view_type=sm"
            news_page = requests.get(listurl)
            news_listsoup = BeautifulSoup(news_page.text, "html.parser")

            # find news list html 
            news_list = news_listsoup.find('ul',{'class' : 'type2'}).find_all('li')

            for news in news_list:
                posting_date = news.find('span',{'class' : 'byline'}).find_all('em')[2].get_text().replace('.','-')
                str_date = posting_date[0:10]
                posted_date = datetime.strptime(str_date, '%Y-%m-%d').date() # "YYYY-MM-DD"
                
                # if news posted date is included in period, then scrap
                if posted_date >= self.start and posted_date <= self.end :
                    main_title = news.find('h4', {'class' : 'titles'}).get_text() # main title
                    
                    # titlelink = news.find('a')['href']
                    # mainurl = "https://www.foodnews.co.kr" + titlelink
                    # mainpage = requests.get(mainurl)
                    # mainpage_soup = BeautifulSoup(mainpage.text, 'html.parser')
                    
                    # # main content
                    # main_content = mainpage_soup.find('div', {'class' : 'sticky-article'}).find_all('p')[:-1]  
                    
                    # # combine comma in list
                    # main_text = []
                    # for content_text in main_content :
                    #     i = content_text.get_text()
                    #     main_text.append(i)
                    # main_text = ' '.join(main_text)
                    
                    # add to dataframe
                    foodjournal_df.loc[idx] = [posted_date, posting_date, main_title, source]
                    idx += 1
                
                # end loop
                elif posted_date < self.start :
                    print("식품저널뉴스 수집완료!")
                    return foodjournal_df
            pagenum += 1

    


    def foodbeverage(self):
        # make dataframe
        foodbeverage_df = pd.DataFrame(columns=("posted_date","posted_date(time)","title","source"))
        idx=0
        pagenum = 1
        source = '식품음료신문'
        # exploration from 1 page
        while True :
            print(pagenum)

            # call 1page url
            listurl = "http://www.thinkfood.co.kr/news/articleList.html?page=" + str(pagenum) + "&total=87428&sc_section_code=&sc_sub_section_code=&sc_serial_code=&sc_area=&sc_level=&sc_article_type=&sc_view_level=&sc_sdate=&sc_edate=&sc_serial_number=&sc_word=&sc_word2=&sc_andor=&sc_order_by=E&view_type=sm&sc_multi_code="
            news_page = requests.get(listurl)
            news_listsoup = BeautifulSoup(news_page.text, "html.parser")

            # find news list html 
            news_list = news_listsoup.find_all('div',{'class' : 'list-block'})

            for news in news_list:
                posting_date = news.find('div',{'class' : 'list-dated'}).get_text()[-16:] # posted timedate
                str_date = posting_date[0:10]
                posted_date = datetime.strptime(str_date, '%Y-%m-%d').date() # "YYYY-MM-DD"

                
                # if news posted date is included in period, then scrap
                if posted_date >= self.start and posted_date <= self.end :
                    main_title = news.find('div',{'class' : 'list-titles'}).find('strong').get_text() # main title
                    
                    # titlelink = news.find('div',{'class' : 'list-titles'}).find('a')['href']
                    # mainurl = "http://www.thinkfood.co.kr/" + titlelink
                    # mainpage = requests.get(mainurl)
                    # mainpage_soup = BeautifulSoup(mainpage.text, 'html.parser')
                    
                    # # main content
                    # main_content = mainpage_soup.find('div', {'id' : 'article-view-content-div'}).find_all('p')
                    
                    # # combine comma in list
                    # main_text = []
                    # for content_text in main_content :
                    #     i = content_text.get_text()
                    #     main_text.append(i)
                    # main_text = ' '.join(main_text)
                    
                    # add to dataframe
                    foodbeverage_df.loc[idx] = [posted_date, posting_date, main_title, source]
                    idx += 1
                
                # end loop
                elif posted_date < self.start :
                    print("식품음료신문 수집완료!")
                    return foodbeverage_df
            pagenum += 1

    def foodtoday(self):
        # make dataframe
        foodtoday_df = pd.DataFrame(columns=("posted_date","posted_date(time)","title","source"))
        idx=0
        pagenum = 1
        source = '식품산업경제뉴스'
        # exploration from 1 page
        while True :
            print(pagenum)

            # call 1page url
            listurl = "http://foodtoday.or.kr/news/article_list_all.html?page=" + str(pagenum)
            news_page = requests.get(listurl)
            news_listsoup = BeautifulSoup(news_page.text, "html.parser")

            # find news list html 
            news_list = news_listsoup.find('ul',{'class' : 'art_list_all'}).find_all('a')
            for news in news_list:

                posting_date = news.find('li',{'class' : 'date'}).get_text() # posted timedate
                str_date = posting_date[0:10]
                posted_date = datetime.strptime(str_date, '%Y-%m-%d').date() # "YYYY-MM-DD"
                
                # if news posted date is included in period, then scrap
                if posted_date >= self.start and posted_date <= self.end :
                    # titlelink = news['href']
                    # mainurl = "http://foodtoday.or.kr/" + titlelink
                    # mainpage = requests.get(mainurl)
                    # mainpage_soup = BeautifulSoup(mainpage.text, 'html.parser')
                    
                    main_title = news.find('h2',{'class' : 'clamp c2'}).get_text() # main title
                    # # main content
                    # main_content = mainpage_soup.find('div', {'id' : 'news_body_area'}).find_all('p')
                    # # combine comma in list
                    # main_text = []
                    # for content_text in main_content :
                    #     i = content_text.get_text()
                    #     main_text.append(i)
                    # main_text = ' '.join(main_text)
                    # # add to dataframe
                    foodtoday_df.loc[idx] = [posted_date, posting_date, main_title, source]
                    idx += 1

                # end loop
                elif posted_date < self.start :
                    print("식품산업경제뉴스 수집완료!")
                    return foodtoday_df
            pagenum += 1

    def naverfood(self):
        # make dataframe
        naverfood_df = pd.DataFrame(columns=("posted_date","posted_date(time)","title","source"))
        idx=0
        source = '식품관련네이버뉴스'

        dt_index = pd.date_range(start=self.start, end=self.end)
        for date in dt_index:
            datenum = date.strftime("%Y%m%d")
            date = datetime.strftime(date,'%Y-%m-%d')
            page_num = 1
            while True :
                
                listurl = "https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid2=238&sid1=103&date="+ datenum +"&page=" + str(page_num)
                
                # call 1page url
                user_agent = {'User-agent': 'Mozilla/5.0'}
                news_page = requests.get(listurl, headers=user_agent)
                news_listsoup = BeautifulSoup(news_page.text, "html.parser")

                # find news list html 
                news_list = news_listsoup.find('div',{'class' : 'list_body newsflash_body'}).find_all('li')
                for news in news_list:
                    # check photo is exist
                    find_dt = news.find_all('dt')
                    if len(find_dt) == 2 :
                        main_title = news.find_all('a')[1].get_text()
                    else :
                        main_title = news.find('a').get_text()

                    titlelink = news.find_all('a')[0]['href']
                    source2 = news.find('span', {'class' : 'writing'}).get_text()
                    real_source = source + "(" + source2 + ")"
                    
                    mainurl = titlelink
                    mainpage = requests.get(mainurl, headers=user_agent)
                    mainpage_soup = BeautifulSoup(mainpage.text, 'html.parser')
                    try :
                        posting_date = mainpage_soup.find('div', {'class' : 'media_end_head_info_datestamp_bunch'}).find('span')['data-date-time'][:16]
                        post_date = posting_date[:10]
                        # main content
                        # main_content = mainpage_soup.find('div', {'id' : 'dic_area'}).get_text()
                        # add to dataframe
                        if post_date == date:
                            posted_date = datetime.strptime(post_date, '%Y-%m-%d').date()
                            naverfood_df.loc[idx] = [posted_date, posting_date, main_title, real_source]
                            idx += 1
                    except AttributeError:
                        print(main_title)
                        pass
                
                strongnum = news_listsoup.find('div', {'class' : 'paging'}).find('strong').get_text()

                page_num += 1
                nexturl = "https://news.naver.com/main/list.naver?mode=LS2D&mid=shm&sid2=238&sid1=103&date="+ datenum +"&page=" + str(page_num)
                next_page = requests.get(nexturl, headers=user_agent)
                next_pagesoup = BeautifulSoup(next_page.text, "html.parser")
                next_strongnum = next_pagesoup.find('div', {'class' : 'paging'}).find('strong').get_text()
                print(strongnum, date)
                if strongnum == next_strongnum :
                    break

        print('네이버뉴스 수집완료!')            
        return naverfood_df
    
    # 뉴스 제목, 본문 수집
    def notnaverfood(self):
        # make dataframe
        notnaverfood_df = pd.DataFrame(columns=("posted_date","posted_date(time)","title","source"))
        idx=0
        source = '식품x네이버뉴스'
        oid_list = ['055'] # 055-sbs , 056-kbs , 020-동아일보

        dt_index = pd.date_range(start=self.start, end=self.end)
        
        for oidnum in oid_list:
            
            for date in dt_index:
                datenum = date.strftime("%Y%m%d")
                date = datetime.strftime(date,'%Y-%m-%d')
                page_num = 1
                while True :
                    listurl = "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=" + oidnum + "&date=" + datenum + "&page=" + str(page_num)
                    
                    # call 1page url
                    user_agent = {'User-agent': 'Mozilla/5.0'}
                    news_page = requests.get(listurl, headers=user_agent)
                    news_listsoup = BeautifulSoup(news_page.text, "html.parser")
                    # find news list html 
                    news_list = news_listsoup.find('div',{'class' : 'list_body newsflash_body'}).find_all('li')
                    for news in news_list:
                        # check photo is exist
                        find_dt = news.find_all('dt')
                        if len(find_dt) == 2 :
                            main_title = news.find_all('a')[1].get_text()
                        else :
                            main_title = news.find('a').get_text()

                        titlelink = news.find_all('a')[0]['href']
                        source2 = news.find('span', {'class' : 'writing'}).get_text()
                        real_source = source + "(" + source2 + ")"

                        mainurl = titlelink
                        mainpage = requests.get(mainurl, headers=user_agent)
                        mainpage_soup = BeautifulSoup(mainpage.text, 'html.parser')
                        
                        if mainpage_soup.find('div', {'class' : 'info'}) == None :
                            try :
                                posting_date = mainpage_soup.find('div', {'class' : 'media_end_head_info_datestamp_bunch'}).find('span')['data-date-time'][:16]
                                post_date = posting_date[:10]
                                # main content
                                # main_content = mainpage_soup.find('div', {'id' : 'dic_area'}).get_text()
                                # add to dataframe
                                if post_date == date:
                                    posted_date = datetime.strptime(post_date, '%Y-%m-%d').date()
                                    notnaverfood_df.loc[idx] = [posted_date, posting_date, main_title, real_source]
                                    idx += 1
                            except AttributeError :
                                print(main_title)
                                pass
                        else:
                            posting_date = mainpage_soup.find('div', {'class' : 'info'}).find('span').get_text()[:15]
                            post_date = posting_date[5:].replace('.', '-')
                            # main content
                            # main_content = mainpage_soup.find('div', {'id' : 'newsEndContents'}).get_text()
                            # add to dataframe
                            if post_date == date:
                                posted_date = datetime.strptime(post_date, '%Y-%m-%d').date()
                                notnaverfood_df.loc[idx] = [posted_date, posting_date, main_title, real_source]
                                idx += 1
                    try:
                        strongnum = news_listsoup.find('div', {'class' : 'paging'}).find('strong').get_text()

                        page_num += 1
                        nexturl = "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=" + oidnum + "&date=" + datenum + "&page=" + str(page_num)
                        next_page = requests.get(nexturl, headers=user_agent)
                        next_pagesoup = BeautifulSoup(next_page.text, "html.parser")
                        next_strongnum = next_pagesoup.find('div', {'class' : 'paging'}).find('strong').get_text()
                        print(strongnum, date)
                        if strongnum == next_strongnum :
                            break
                    except AttributeError :
                        print(main_title)
                        pass            
        
        print('네이버뉴스 수집완료!')            
        return notnaverfood_df   
    
    # 제목만 수집
    def notnaverfood_title(self):
        # make dataframe
        notnaverfood_df = pd.DataFrame(columns=("title","source"))
        idx=0
        source = '식품x네이버뉴스'
        oid_list = ['032','005','081','055','056','020','022','023','025','028','469','437','214','448'] # 055-sbs , 056-kbs , 020-동아일보

        dt_index = pd.date_range(start=self.start, end=self.end)

        for oidnum in oid_list:
            # 날짜 반복
            for date in tqdm(dt_index):
                datenum = date.strftime("%Y%m%d")
                date = datetime.strftime(date,'%Y-%m-%d')
                page_num = 1
                
                # pagenum 반복
                while True :
                    listurl = "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=" + oidnum + "&date=" + datenum + "&page=" + str(page_num)
                    
                    # call 1page url
                    user_agent = {'User-agent': 'Mozilla/5.0'}
                    news_page = requests.get(listurl, headers=user_agent)
                    news_listsoup = BeautifulSoup(news_page.text, "html.parser")
                    # find news list html 
                    news_list = news_listsoup.find('div',{'class' : 'list_body newsflash_body'}).find_all('li')
                    for news in news_list:
                        # check photo is exist
                        find_dt = news.find_all('dt')
                        if len(find_dt) == 2 :
                            main_title = news.find_all('a')[1].get_text()
                        else :
                            main_title = news.find('a').get_text()
                        notnaverfood_df.loc[idx] = [main_title, source]
                        idx += 1
                    try:
                        strongnum = news_listsoup.find('div', {'class' : 'paging'}).find('strong').get_text() # 현재 pagenum

                        page_num += 1 
                        nexturl = "https://news.naver.com/main/list.naver?mode=LPOD&mid=sec&oid=" + oidnum + "&date=" + datenum + "&page=" + str(page_num)
                        next_page = requests.get(nexturl, headers=user_agent)
                        next_pagesoup = BeautifulSoup(next_page.text, "html.parser")
                        next_strongnum = next_pagesoup.find('div', {'class' : 'paging'}).find('strong').get_text()

                        if strongnum == next_strongnum : # 다음 pagenum을 불러왔는데 현재 pagenum와 같으면 while문 빠져나가서 다음 날짜로 넘어가기
                            break
                    except AttributeError :
                        print(listurl)
                        break
        
        
        print('네이버뉴스 제목 수집완료!')            
        return notnaverfood_df
    
    # 식품안전나라 국내뉴스 수집
    def foodsafety_news(self):
        foodsafe_df = pd.DataFrame(columns=('title', 'content', 'date'))
        df_date = pd.DataFrame(columns=("menu", "date"))
        idx=0

        path = '.\chromedriver.exe'
        driver = webdriver.Chrome(path)

        # 국내뉴스
        url = 'https://www.foodsafetykorea.go.kr/portal/board/board.do?menu_grp=MENU_NEW05&menu_no=2859'
        driver.get(url)
        driver.implicitly_wait(3)

        # 제목리스트
        #listFrame
        newslist = driver.find_elements(By.CSS_SELECTOR, "#listFrame > tr")
        print(len(newslist))
        # driver.find_element(By.CSS_SELECTOR, "#div_ctgType01_0").click()
        # driver.implicitly_wait(3)
        # date = driver.find_element(By.CSS_SELECTOR, "#listFrame > a:nth-child(1) > ul > li.date").text
        # driver.implicitly_wait(3)
        # df_date.loc[idx] = [menu,date]
        # idx += 1
        # print(df_date)
        