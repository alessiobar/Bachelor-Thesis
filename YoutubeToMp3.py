import youtube_dl, os
from youtubesearchpython import VideosSearch
import pandas as pd

df = pd.read_excel('outttt.xlsx')
os.chdir("./Tracks")

links=[]
ll=[]
for x in range(len(df["Song_ID"])):
    ll.append(df["Song"][x]+" - "+df["Artist"][x])
ll=list(set(ll)) #to remove double entries

for x in ll:
    videosSearch = VideosSearch(x, limit = 1)
    try:
        lk=videosSearch.result()["result"][0]["link"]
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': "{} - {}.{}".format(x[:x.find("-")-1], x[x.find("-")+2:],"mp3"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }]}
        youtube_dl.YoutubeDL(ydl_opts).download([lk])

    except Exception:
        print(x)
