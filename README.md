# Song-Popularity-Prediction
This repository contains the code of my Bachelor's thesis, titled "*Prediction of song popularity with a textual and audio based approach*", and available for download at http://tesi.luiss.it/31943/ (alternatively see `Thesis.pdf`).

**TL;DR** This thesis proposes a blended approach for predicting song popularity combining some textual approaches presented in Berger et al. (2018; 2020), with an audio based one suggested in Lee et al. (2018).

For a more extensive description refer to the thesis document.

## Scope

The main objective is to verify the propositions presented in Berger et al. (2018; 2020) by testing them for statistical significance, and to be able to generalize their approach to any new data. Hence i will reproduce their analysis from scratch (starting from data collection). 

Then, i will extend the analysis by including some audio-based features described in Lee et al. (2018).

## Data

The original dataset from Berger et al. contains data about 4200 songs and it was scraped from *Billboard.com* (see *You_S1_Data_NoBillboardRanking.csv* at https://osf.io/cbguq). In the specific, the authors scraped the weekly most popular downloaded songs chart (top-50), quarterly for 3 years, and for 7 major genres (i.e., christian, country, dance, rock, pop, rap, r&b).

To aquire the *Billboard*'s data refer to `BillboardScraper.py`, which returns the following columns: *song name, artist name, genre, date and rank*. However notice that, as of now (2023) and unlike few years ago, the *Digital Song Sales charts* are only accessible upon a monthly payment. The song lyrics were added instead using `AddLyrics.py`.

The audio tracks, needed for extending the anlysis, were downloaded in mp3 format from *Youtube.com* (see `YoutubeToMp3.py`).

## Feature Engineering
Most of the variables needed by the models had to be feature engineered, especially the ones for audio data.

Textual Analysis Features:

top 100 parole 
- *Linguistic Inquiry and Word Count* (*LIWC*) 2015 was used for extracting Second Person Pronouns and many other word metrics directly from lyrics.

- *Latent Dirichlet Allocation* (*LDA*) was performed on lyrics (see `LDA.r`), after some preprocessing (see `preprocessLDA.py`, to define 10 topics and word distribution per topic. (see `aooooooooo.py`). 
- A costumized version of the *Linguistic Style Matching* (*LSM*) equation was computed, (customizing the original equation to make it measure difference rather than similarity)

Audio Analysis Features:

- The *Structural Change* of two *Complexity features* computed from scratch (ie. Chroma and Timbre) was calculated as described in Lee et al. (see `aooooooooo.py`). 

- *MFCC* and *Arousal* features were added as described in Lee et al. (see `aooooooooo.py`). 

## Analysis

RQ1: The statistical significance of the feature engineered variables for the textual analysis was checked using an Ordinary Least Squares linear (OLS) regression, reproducing 10 models described in Berger et al. (see also section 4.3 of my thesis). For the audio analysis instead, 5 models from the Lee et al. paper were reproduced.

RQ2: Three Random Forests and three Support Vector Machines models were built for predicting song’s rank, tuning the hyperparameters using GridSearchCV and selecting the best one in terms of performances (MSE? MAE? che ho usato?)

## Results
...

## (Few) References
- Berger, Jonah and Grant Packard (2020), “Thinking of You: How Second-Person Pronouns Shape Cultural Success”
- Berger, Jonah and Grant Packard (2018), “Are Atypical Things More Popular?”
- Lee, Jong-Seok and Junghyuk Lee (2018), “Music Popularity: Metrics, Characteristics, and Audio-Based Prediction”

.

PS se non voglio che si vedano i commit, posso crearne diretto un altra di repo, questa la lascio privata al limite o la cancello

The dataset is made of 161 columns, including: *id, date, genre, song, artist, LDAtopics, *

storytelling



(..For the **text mining** part, i will rebuild t)





Since two different types of analyses are carried out, ie. Text Mining and Audio Mining, two datasets will be used: 
- The first contains song-related information about 4200 songs, it was made available by Berger et al. (see *You_S1_Data_NoBillboardRanking.csv* at https://osf.io/cbguq), and it was scraped originally from *Billboard.com*. However, the lyrics were not present and were added using `AddLyrics.py`. (Refer instead to `BillboardScraper.py` to acquire any new/different data; however notice that, as of now (2022), the *Digital Song Sales charts* are accessible only by paying a monthly fee).
- The other one contains the corresponding audio tracks, downloaded in mp3 format from *Youtube.com* (see `YoutubeToMp3.py`)
