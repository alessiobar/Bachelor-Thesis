"""
This script builds the Billboard URLs from scratch and scrapes them sequentially taking the fields of interest (ie. songs, artists, rank, genre).

Warning: this was made in 2021, as for any other web scraping script, since websites continuosly change, it may have to be fixed in the future!
"""

import requests, os, re
import pandas as pd
from bs4 import BeautifulSoup

#To have comparable results, let's take the same weeks that Berger et al. took for their analyses
df_temp = pd.read_excel("You_S1_Data_NoBillboardRanking.xlsx")
dates = list(df_temp.groupby(['date'])['date'].unique().index.strftime('%Y-%m-%d'))

def urlFinder():
    l = ["https://www.billboard.com/charts/christian-digital-song-sales", "https://www.billboard.com/charts/country-digital-song-sales",
         "https://www.billboard.com/charts/dance-electronic-digital-song-sales", "https://www.billboard.com/charts/r-and-b-digital-song-sales",
         "https://www.billboard.com/charts/pop-digital-song-sales", "https://www.billboard.com/charts/rock-digital-song-sales",
         "https://www.billboard.com/charts/rap-digital-song-sales"]
    ll=[]
    for fixd in l:
        for date in dates:
            ll.append(fixd + "/" + date)
    return ll

def webScraper(url, genre):
    response = requests.get(url=url)
    ranks, songs, artists = [], [], []
    if not response.ok: 
        return "No"
    soup = BeautifulSoup(response.content, 'html.parser')
    wpage = soup.prettify().split("\n")
    n = ["featuring", "&amp;", "+", " X ", " x "]
    i=0
    for x in wpage:
        if "chart-list-item__rank" in x:
            rank = int(wpage[i + 1][wpage[i + 1].find([x for x in wpage[i + 1] if x!=" "][0]):])
            ranks.append(rank)
        if "chart-list-item__title-text" in x:
            song= wpage[i + 1][wpage[i + 1].find([x for x in wpage[i + 1] if x!=" "][0]):]
            for x in n:
                while x in song:
                    j=song.find(x)
                    song=song[:j-1]+","+song[j+len(x):]
            songs.append(song)
        if "chart-list-item__artist" in x:
            artist= wpage[i + 1][wpage[i + 1].find([x for x in wpage[i + 1] if x!=" "][0]):].lower()
            if artist[:2]=="<a":
                artist=re.sub("-"," ",artist[16:-2])
            for x in n:
                while x in artist:
                    if x!=" X " or x!=" x ":
                        j=artist.find(x)
                        artist=artist[:j-1]+","+artist[j+len(x):]
                    else:
                        j = artist.find(x)
                        artist = artist[:j - 1] + ", " + artist[j + len(x):]
            artists.append(artist)
        i+=1
    df = pd.DataFrame(
        {"Song": songs, "Artist": artists, "Rank":ranks, "Genre":genre})
    return df

urls = urlFinder()

#To Repeat a few times since Captcha exists
dfs = []
for x in urls:
    genre = x[len('https://www.billboard.com/charts/'):-len("/20??-??-??")] #MA CHE È STA ROBA OH? /22
    r = webScraper(x, genre)
    if isinstance(r, pd.DataFrame):
        dfs.append(r)
        urls.remove(x)
    else:
        break
print("remaining: " + str(len(urls)))

#urls = [x[:len('https://www.billboard.com/charts/rap-streaming-song')] + "s" + x[len('https://www.billboard.com/charts/rap-streaming-song'):] for x in urls]
out = pd.concat([x for x in dfs])
