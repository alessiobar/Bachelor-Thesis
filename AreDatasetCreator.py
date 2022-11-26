# Module to Recreate ARE dataset (inserting lyrics into YOU dataset, but which one?!)
import lyricsgenius, os
import pandas as pd

genius = lyricsgenius.Genius("IOM90P4leGH-HxQbeNKozTck_pfXfxMNEUZ7tQ05WXLYOOfWX8P5tx1CX5VsXu3p")
os.chdir(r"C:\Users\alema\Desktop")
df = pd.read_excel("Thesis\You_S1_Data_NoBillboardRanking.xlsx")

#df["Lyrics"] = 0
def lyricFinder(title, auth):
    print(auth)
    song = genius.search_song(title, auth)
    return song.lyrics

def stringNomalizer(title):
    if title.find("(")!=-1:
        title = title[:title.find("(")-1]
    return title

for x in range(1667, len(df["song"])): #togli il primo numero
    df.song[x] = stringNomalizer(str(df.song[x]))
    try:
        df["Lyrics"][x] = lyricFinder(str(df.song[x]), str(df.artist[x]))
    except Exception:
        print("zxbmxcvmnxcvxc")
        try:
            df["Lyrics"][x] = lyricFinder(str(df.song[x]), str(df.artist[x]))
        except Exception:
            print("---")
            print(df.song[x])
            print(df.artist[x])
            print("NO")

#df.to_excel("outdataset1.xlsx")




