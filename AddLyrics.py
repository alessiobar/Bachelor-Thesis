import lyricsgenius, os
import pandas as pd

genius = lyricsgenius.Genius(" < your token here >")
df = pd.read_excel("You_S1_Data_NoBillboardRanking.xlsx")
df["Lyrics"] = 0

def lyricFinder(title, auth):
    print(auth)
    song = genius.search_song(title, auth)
    return song.lyrics

def stringNomalizer(title):
    if title.find("(")!=-1:
        title = title[:title.find("(")-1]
    return title

for x in range(len(df["song"])):
    df.song[x] = stringNomalizer(str(df.song[x]))
    try:
        df["Lyrics"][x] = lyricFinder(str(df.song[x]), str(df.artist[x]))
    except Exception:
        print("zxbmxcvmnxcvxc")
        try:
            df["Lyrics"][x] = lyricFinder(str(df.song[x]), str(df.artist[x]))
        except Exception:
            print(" :( ")
            print(df.song[x])
            print(df.artist[x])
            
df.to_excel("outdataset1.xlsx")
