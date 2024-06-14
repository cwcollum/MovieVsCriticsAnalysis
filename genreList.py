import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

movie_db = pd.read_csv('Movie Database - Sorted Movie Data.csv')

#Empty lists for counting single genres and their counts
singleGenre = dict()

#Empty lists for counting genres as groups (to be sorted alphabetically for consistency)
groupGenre = dict()

# Empty dict for counting the genres by primary genre specifically
primoGenre = dict()

for index, row in movie_db.iterrows():
    genreList = re.split('[,]\s', row['genres'])
    if(genreList[0] not in primoGenre):
        primoGenre[genreList[0]] = 1
    else:
        primoGenre[genreList[0]] += 1
    genreList.sort()
    size = len(genreList)
    if(size > 2):
        for i in range(0, size - 2):
            genreList.pop()
    print(genreList)
    for genre in genreList:
        if(genre not in singleGenre):
            singleGenre[genre] = 1
        else:
            singleGenre[genre] += 1
    genreList = ''.join(genreList)
    if(genreList not in groupGenre):
        groupGenre[genreList] = 1
    else:
        groupGenre[genreList] += 1

sinKeys = list(singleGenre.keys())
sinVal = list(singleGenre.values())
sortValIndex = np.argsort(sinVal)
sortSingleGenre = {sinKeys[i]: sinVal[i] for i in sortValIndex}

grpKeys = list(groupGenre.keys())
grpVal = list(groupGenre.values())
sortVal2Index = np.argsort(grpVal)
sortGroupGenre = {grpKeys[i]: grpVal[i] for i in sortVal2Index}
# print(sortSingleGenre)
# print(groupGenre)

priKeys = list(primoGenre.keys())
priVal = list(primoGenre.values())
sortVal3Index = np.argsort(priVal)
sortPrimeGenre = {priKeys[i]: priVal[i] for i in sortVal3Index}

plt.figure(figsize=(20, 10))
plt.xticks(rotation='vertical')
plt.bar(sortSingleGenre.keys(), sortSingleGenre.values())
plt.show()

plt.figure(figsize=(20, 10))
plt.xticks(rotation='vertical')
plt.bar(sortGroupGenre.keys(), sortGroupGenre.values())
plt.show()

plt.figure(figsize=(20, 10))
plt.xticks(rotation='vertical')
plt.bar(sortPrimeGenre.keys(), sortPrimeGenre.values())
plt.show()