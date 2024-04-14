from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import os

movie_list = pd.read_csv('movie_list.csv', names=['Movie Name'])

for index, title in movie_list.iterrows():
    df = pd.DataFrame(columns=['Dates', 'Scores', 'Reviews'])
    movie_title = title['Movie Name']

    url = 'https://www.rottentomatoes.com/m/'+movie_title+'/reviews?type=user'
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}
    request = Request(url, headers=headers)

    page = urlopen(request, timeout=3)
    html = page.read().decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    folder_path = movie_title
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    api_request = soup.find_all('load-more-manager')
    endpoint = 'https://www.rottentomatoes.com'+api_request[0]['endpoint']
    endcursor = api_request[0]['endcursor']
    r = requests.get(endpoint)
    data = r.json()
    for sample in data['reviews']:
        score = ""
        if(sample['rating'] > 3):
            score = "positive"
        elif(sample['rating'] < 3):
            score = "negative"
        else:
            score = "neutral"
        df.loc[len(df.index)] = [sample['creationDate'], score, sample['quote']]

    for i in range(100):
        if(data['pageInfo']['hasNextPage'] == True):
            endcursor = data['pageInfo']['endCursor']
            r = requests.get(endpoint + "?after=" + endcursor + "&pageCount=100")
            data = r.json()
            time.sleep(0.2)
            for sample in data['reviews']:
                score = ""
                if(sample['rating'] > 2.5):
                    score = "positive"
                elif(sample['rating'] < 2.5):
                    score = "negative"
                else:
                    score = "neutral"
                df.loc[len(df.index)] = [sample['creationDate'], score, sample['quote']]
            print('Request ', i, ' succeeded')
        else:
            break

    df.to_csv(movie_title+'/'+'a_'+movie_title+'.csv', encoding='utf-8', index=False)