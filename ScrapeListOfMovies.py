# CURRENTLY NOT IN USE


# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By

# from bs4 import BeautifulSoup
# from random import randint
# import time

# brave_path = "C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe"

# option = webdriver.ChromeOptions()
# option.binary_location = brave_path

# driver = webdriver.Chrome(options=option)
# driver.get('https://www.rottentomatoes.com/browse/movies_at_home/')

# # click the button exactly 8 times to load more movies
# for n in range(8):
#     button = driver.find_element(By.CLASS_NAME, "discovery__actions").find_element(By.TAG_NAME, "button")
#     button.click()
#     # ... we told it which button to click
#     # make a random wait time between 1 and 10 seconds to look less bot-like
#     s = randint(1, 5)
#     # sleep that number of seconds
#     time.sleep(s)

# # page_source is a variable created by Selenium - it holds all the HTML
# page = driver.page_source

# soup = BeautifulSoup(page, "html.parser")

# # each tile contains all info for one movie
# tiles = soup.find_all(class_="js-tile-link")

# movie_titles = []

# for tile in tiles:
#     span = tile.find('span', class_="p--small")
#     title = span.text.strip()
#     if len(title) > 0:
#         movie_titles.append( title )
#         print( title )

# print("There are " + str( len(movie_titles) ) + " movies in the list.")

# driver.quit()

from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import time

review_url = "https://thewolfmancometh.com/2012/05/09/the-avengers-2012-review/"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36'}
request = Request(review_url, headers=headers)
review_page = urlopen(request)
n_html = review_page.read().decode("utf-8")
n_soup = BeautifulSoup(n_html, "html.parser")

for text in n_soup.find_all('p'):
  punct = text.get_text()[-2:]
  if('.' or '!' or '?' in punct):
    if(len(text.get_text().split()) >= 25):
        print(text.get_text())