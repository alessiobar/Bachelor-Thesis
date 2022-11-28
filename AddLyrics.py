"""
This script adds lyrics to the existing Berger et al.'s dataset using the Genius API
"""

import lyricsgenius, os, time, pickle
import pandas as pd

genius = lyricsgenius.Genius(" < your token here >")
df = pd.read_excel("You_S1_Data_NoBillboardRanking.xlsx")
df["Lyrics"] = 0

def lyricFinder(title, auth):
    #print(auth)
    song = genius.search_song(title, auth)
    return song.lyrics

def stringNomalizer(title):
    if title.find("(")!=-1:
        title = title[:title.find("(")-1]
    return title

toDoubleCheck = []
for x in range(len(df["song"])):
    #print(x)
    df.loc[x, "song"] = stringNomalizer(str(df.song[x]))
    try:
        lyrics = lyricFinder(str(df.song[x]), str(df.artist[x]))
        if len(lyrics.split()) - df["WC"][x] > 100:
            toDoubleCheck.append((x, df.song[x], df.artist[x]))
        else: df.loc[x, "Lyrics"] = lyrics

    except Exception as e: #mainly HTTPSConnectionPool erros 
        print(e)
        triedTimes = 0
        while triedTimes < 5:
            if triedTimes > 3: time.sleep(5)
            try:
                df.loc[x, "Lyrics"] = lyricFinder(str(df.song[x]), str(df.artist[x]))
                break
            except Exception:
                pass
            triedTimes += 1

df.to_excel("outdataset1(2022).xlsx")
with open('toDoubleCheck.pkl', 'wb') as f:
    pickle.dump(toDoubleCheck, f)
