# Song-Popularity-Prediction
This repository contains the code of my Bachelor's thesis, titled "*Prediction of song popularity with a textual and audio based approach*", and available for download at http://tesi.luiss.it/31943/.

The thesis proposes a blended approach for predicting song popularity combining some textual based approaches presented in Berger et al. (2018; 2020), with an audio based one suggested in Lee et al. (2018).

## Data
Since two different types of analysis are carried out, ie. Text Mining and Audio Mining, two datasets are used: 
- The first contains song-related information of 4200 songs, it was made available by Berger et al. (see *You_S1_Data_NoBillboardRanking.csv* at https://osf.io/cbguq), and it was scraped originally from *Billboard.com*. Although, the lyrics were not present and were added them using `AreDatasetCreator.py` (refer instead to `BillboardScraper.py` to acquire any new/different data; however notice that, as of now (2022), the *Digital Song Sales charts* are accessible only by paying a monthly fee).
- The other one contains the corresponding audio tracks, downloaded in mp3 format from *Youtube.com* (see `YoutubeToMp3.py`)

## Feature Engineering



### Billboard Scraping


- one with lyrics and other song-related features, scraped from seven major genres of *Billboard’s digital download rankings*
- another one with the corresponding audio tracks, downloaded in mp3 format from *Youtube.com*

in tutto ciò nella tesi c'era anche R loooool
