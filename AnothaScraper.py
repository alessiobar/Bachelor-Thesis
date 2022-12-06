import requests, os, re, lyricsgenius
import pandas as pd
from bs4 import BeautifulSoup

#os.chdir(r"C:\Users\alema\Desktop")
genius = lyricsgenius.Genius("IOM90P4leGH-HxQbeNKozTck_pfXfxMNEUZ7tQ05WXLYOOfWX8P5tx1CX5VsXu3p")
dates = [x.strftime('%Y-%m-%d') for x in list(set(list(pd.read_excel("outdataset1.xlsx")["date"])))]

def urlFinder():
    l = ["https://www.billboard.com/charts/christian-digital-song-sales", "https://www.billboard.com/charts/country-digital-song-sales",
         "https://www.billboard.com/charts/dance-electronic-digital-song-sales", "https://www.billboard.com/charts/r-and-b-digital-song-sales",
         "https://www.billboard.com/charts/pop-digital-song-sales", "https://www.billboard.com/charts/rock-digital-song-sales",
         "https://www.billboard.com/charts/rap-digital-song-sales"] #quelle del pop non esistono di streaming bah
    ll=[]
    for fixd in l:
        for date in dates:
            ll.append(fixd + "/" + date)
    return ll

def webScraper(url, genre):
    response = requests.get(url=url)
    ranks, songs, artists, lastWeeks, peaks, wksOnCharts = [], [], [], [], [], []
    if not response.ok:
        return "No"

    soup = BeautifulSoup(response.content, 'html.parser')
    wpage = soup.prettify().split("\n")

    n = ["featuring", "&amp;","+", " X ", " x "]

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
    #print(df)
    return df

#out = pd.concat([webScraper(x) for x in urlFinder()])
urls = urlFinder()
dfs= []



#da fare n mila volte :DDDDDDDDDD perché ci sta il Captcha che ti stoppa, GL :)
for x in urls:
    genre = x[len('https://www.billboard.com/charts/'):-len("/20??-??-??")]
    r = webScraper(x, genre)
    if isinstance(r, pd.DataFrame):
        dfs.append(r)
        urls.remove(x)
    else:
        break
print("remaining: " + str(len(urls)))

#urls = [x[:len('https://www.billboard.com/charts/rap-streaming-song')] + "s" + x[len('https://www.billboard.com/charts/rap-streaming-song'):] for x in urls]
out = pd.concat([x for x in dfs])