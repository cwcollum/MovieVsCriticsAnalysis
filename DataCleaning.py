import os
import pandas as pd
import langdetect as ld
import re

def remove_urls(text):
    pattern = re.compile(r'http\S+|www\S+')
    return pattern.sub(r'', text)

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def cleaning(df):
    df['Reviews'] = df['Reviews'].str.lower()
    df = df.replace('\n', ' ', regex=True)
    df = df.replace('\r', ' ', regex=True)
    df = df.replace('\t', '', regex=True)
    df = df.replace('\s', ' ', regex=True)
    df = df.replace('ð', '', regex=True)
    df = df[df['Reviews'] != '']
    for index, row in df.iterrows():
        try:
            df.loc[index, 'Reviews'] = remove_urls(row['Reviews'])
            df.loc[index, 'Reviews'] = remove_html_tags(row['Reviews'])
            if(ld.detect(row['Reviews']) != 'en'):
                lang = ld.detect_langs(row['Reviews'])
                en = False
                for item in lang:
                    if(item.lang == 'en'):
                        en = True
                if(en == False):
                    df.loc[index, 'Reviews'] = ""
        except:
            df.loc[index, 'Reviews'] = ""
    df = df.replace(r'[^\w\s]|_' , '', regex=True)
    df = df[df['Reviews'] != '']
    return df

dir1 = 'data'

for folder_name in os.listdir(dir1):
    folder = folder_name
    dir2 = dir1+'/'+folder
    for movie in os.listdir(dir2):
        movie_folder = movie
        dir3 = dir2+'/'+movie_folder
        for file in os.listdir(dir3):
            print("Cleaning", file)
            data = pd.read_csv(dir3+'/'+file)
            data = cleaning(data)
            data.to_csv(dir3+'/'+file, encoding='utf-8', index=False)
            print(file, "finished")