from selenium import webdriver
from selenium.webdriver.common.by import By

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd
import random
import time
import os

def scrapeReviewSite(url):
    review = ""
    review_url = url
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}
    try:
        request = Request(review_url, headers=headers)
        review_page = urlopen(request, timeout=3)
        n_html = review_page.read().decode("utf-8")
        n_soup = BeautifulSoup(n_html, "html.parser")
        for text in n_soup.find_all('p'):
            punct = text.get_text()[-2:]
            if('.' in punct or '!' in punct or '?' in punct):
                if(len(text.get_text().split()) >= 25):
                    review = review + " " + text.get_text()
    except:
        print("Website could not be opened")
        return review
    return review

def convertScore(rating):
    score = ''
    rating = rating.replace('\n','')
    rating = rating.replace(' ','')
    rating = rating.replace('|','')
    try:
        rating = rating.split(':',1)[1]
    except:
        score = '0.0'
        return score
    if('/4' in rating or '/4.0' in rating):
        score = float(rating.split('/',1)[0]) * 1.25
    elif('/5' in rating or '/5.0' in rating):
        score = rating.split('/',1)[0]
    elif('/10' in rating or '/10.0' in rating):
        score = float(rating.split('/',1)[0]) / 2
    elif('A+' in rating or 'a+' in rating):
        score = '5'
    elif('A' in rating or 'a' in rating):
        score = '4.75'
    elif('A-' in rating or 'a-' in rating):
        score = '4.5'
    elif('B+' in rating or 'b+' in rating):
        score = '4'
    elif('B' in rating or 'b' in rating):
        score = '3.75'
    elif('B-' in rating or 'b-' in rating):
        score = '3.5'
    elif('C+' in rating or 'c+' in rating):
        score = '3'
    elif('C' in rating or 'c' in rating):
        score = '2.75'
    elif('C-' in rating or 'c-' in rating):
        score = '2.5'
    elif('D+' in rating or 'd+' in rating):
        score = '2'
    elif('D' in rating or 'd' in rating):
        score = '1.75'
    elif('D-' in rating or 'd-' in rating):
        score = '1.5'
    elif('F+' in rating or 'f+' in rating):
        score = '1'
    elif('F' in rating or 'f' in rating):
        score = '0.75'
    elif('F-' in rating or 'f-' in rating):
        score = '0.5'
    else:
        score = '0.0'
    return str(score)

movie_list = pd.read_csv('movie_list.csv', names=['Movie Name'])
brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"
option = webdriver.ChromeOptions()
option.binary_location = brave_path
option.add_argument("user-agent")

for index, title in movie_list.iterrows():
    movie_title = title['Movie Name']
    website = 'https://www.rottentomatoes.com/m/'+movie_title+'/reviews'
    driver = webdriver.Chrome(options=option)
    driver.get(website)

    folder_path = movie_title
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    while True:
        try:
            button = driver.find_element(By.CLASS_NAME, "load-more-container").find_element(By.TAG_NAME, "rt-button")
            button.click()
        except:
            break
        s = random.uniform(0.75, 1.25)
        time.sleep(s)

    time.sleep(2)
    page = BeautifulSoup(driver.page_source, "html.parser")
    reviews = page.find_all(class_="review-text")
    urls = page.find_all(class_="original-score-and-url")
    scores = page.find_all(class_="review-data")

    list_review = []
    list_urls = []
    list_sentiment = []
    list_scores = []
    list_dates = []

    for score in scores:
        if(score.find('score-icon-critic-deprecated')['state'] == 'fresh'):
            list_sentiment.append('positive')
        else:
            list_sentiment.append('negative')

    count = 0
    for url in urls:
        href = url.find('a')
        date = url.find('span')
        site = ""
        try:
            site = href['href']
        except:
            print("No website link")
        list_urls.append(site)
        list_dates.append(date.get_text())
        list_review.append("")
        list_sentiment.append("")
        if("youtu.be" in site or "youtube" in site or site == ""):
            list_review[count] = ""
        else:
            list_review[count] = scrapeReviewSite(site)
        rate = url.contents[2]
        list_scores.append(convertScore(rate))
        if('C' in rate or '2/4' in rate or '5/10' in rate or '2.5/5' in rate):
            list_sentiment[count] = 'neutral'
        if(list_scores[count] == '0.0'):
            if(list_sentiment[count] == 'positive'):
                list_scores[count] = '4.0'
            elif(list_sentiment[count] == 'negative'):
                list_scores[count] = '1.0'
            else:
                list_scores[count] = '2.5'
        count += 1
        print("Review ", count)

    count = 0
    for review in reviews:
        if (list_review[count] == ""):
            list_review[count] = review.get_text()
        count += 1

    driver.quit()

    framework = list(zip(list_dates, list_scores, list_review, list_sentiment))
    df = pd.DataFrame(framework, columns=['Dates', 'Scores', 'Reviews', 'Sentiment'])

    df.to_csv(movie_title+'/'+'c_'+movie_title+'.csv', encoding='utf-8', index=False)