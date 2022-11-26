"""
This script builds the Billboard URLs from scratch, scrapes them sequentially taking the fields of interest
(ie. songs, artists, lastWeeks, peaks, wksOnCharts), finds the lyrics for each song using the Genius API

Warning: as for any other web scraping script, websites continuosly change, hence the script may have to be fixed in the future
"""

import requests, os, lyricsgenius
import pandas as pd
from bs4 import BeautifulSoup

genius = lyricsgenius.Genius("< your token here>") #Genius API (free) Access Token

def urlFinder():
    """Builds by hand the Billboard URLs"""
    fixd = "https://www.billboard.com/charts/hot-100/" #fixed part of the URL
    sat = [] #variable part of the URL
    for year in range(2018, 2021):
        for x in pd.date_range(start=str(year), end=str(year + 1), freq='W-SAT').strftime('%m/%d/%Y').tolist():
            x = x.replace("/","-")
            x = x[-4:] + "-" + x[:5]
            sat.append(x)
    return [fixd + x for x in sat]

def lyricFinder(title, auth):
    """Finds lyrics of a sogn from the song title and the author(s) name"""
    print(auth)
    song = genius.search_song(title, auth)
    return song.lyrics

def webScraper(url):
    """Scrapes from an URL the fields of interest"""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, 'html.parser')
    wpage = soup.prettify().split("\n")

    songs, artists, lastWeeks, peaks, wksOnCharts = [], [], [], [], []
    n = ["Featuring", "&amp;", "+", " X ", " x "]
    i = 0
    for x in wpage:
        if "chart-element__information__song" in x:
            song = wpage[i+1]
            song=song[len("           "):]
            for x in n:
                while x in song:
                    j=song.find(x)
                    song=song[:j-1]+","+song[j+len(x):]
            songs.append(song)
        if "chart-element__information__artist text--truncate color--secondary" in x:
            artist = wpage[i+1]
            artist= artist[len("           "):]
            for x in n:
                while x in artist:
                    if x!=" X " or x!=" x ":
                        j=artist.find(x)
                        artist=artist[:j-1]+","+artist[j+len(x):]
                    else:
                        j = artist.find(x)
                        artist = artist[:j - 1] + ", " + artist[j + len(x):]
            artists.append(artist)
        if "chart-element__information__delta__text text--last" in x:
            lastWeek = wpage[i+1][len("            "):]
            lastWeeks.append(lastWeek[:lastWeek.find(" ")])
        if "chart-element__information__delta__text text--peak" in x:
            peak = wpage[i+1][len("            "):]
            peaks.append(peak[:peak.find(" ")])
        if "chart-element__information__delta__text text--week" in x:
            wksOnChart = wpage[i+1][len("            "):]
            wksOnCharts.append(wksOnChart[:wksOnChart.find(" ")])
        i+=1

    lyrics=[]
    for x, y in zip(songs, artists):
        y = y.split(",") #not tested Ex-Ante fix for lyrics, before it was just y.
        y = y[0] #from my experience, the API works best with a single artist name, with more it may end up selecting a translation for the lyrics in a non original vers.
        lyrics.append(lyricFinder(x,y))
    date = url[41:]
    df = pd.DataFrame({"Song":songs, "Artist":artists, "Last Week":lastWeeks, "Peak":peaks, "Weeks on Chart":wksOnCharts,
                       "Date":date, "Lyrics":lyrics})
    return df

if __name__ == '__main__':
    out = pd.concat([webScraper(x) for x in urlFinder()])
    out.to_excel("out.xlsx")

''' #Ex-Post Fix (for lyrics found in a translated version)
x=0
while x <len(df["Artist"]):
    try: # to avoid --> HTTPSConnectionPool(host='genius.com', port=443): Read timed out. (read timeout=5)
        auth = df["Artist"][x].split(",")
        df["Lyrics"][x] = lyricFinder(df["Song"][x], auth[0])
    except Exception:
        x-=1
    x+=1
'''
