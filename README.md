# Song-Popularity-Prediction
This repository contains the code of my Bachelor's thesis, titled "*Prediction of song popularity with a textual and audio based approach*", and available for download at http://tesi.luiss.it/31943/ (alternatively see `Thesis.pdf`).

**TL;DR** This thesis proposes a blended approach for predicting song popularity combining some textual approaches presented in Berger et al. (2018; 2020), with an audio based one suggested in Lee et al. (2018).

For a more extensive description refer to the thesis document.

## Scope

The main objective is to verify the propositions presented in Berger et al. (2018; 2020) by testing them for statistical significance, and to be able to generalize their approach to any new data. Hence i will reproduce their analysis from scratch (starting from data collection). 
Then, i will extend the analysis by including some audio-based features described in Lee et al. (2018).

## Data

The original dataset from Berger et al. contains data about 4200 songs and was scraped from *Billboard.com* (see *You_S1_Data_NoBillboardRanking.csv* at https://osf.io/cbguq). In the specific, the authors scraped the weekly most popular downloaded songs chart (top-50), quarterly for 3 years, and for 7 major genres (i.e., christian, country, dance, rock, pop, rap, r&b).

To aquire the *Billboard*'s data refer to `BillboardScraper.py`, which returns the following columns: *song name, artist name, genre, date and rank*. However notice that, as of now (2023) and unlike few years ago, the *Digital Song Sales charts* are only accessible upon a monthly payment (sc. Billboard Pro). The song *lyrics* were added instead using `AddLyrics.py`.

The audio tracks, needed to extend the analysis, were downloaded in mp3 format from *Youtube.com* (see `YoutubeToMp3.py`).

## Feature Engineering

Most of the variables had to be feature engineered.

For the **Textual Analysis**, after some preprocessing (which includes *Case Normalization, Tokenization, Stop words removal* and *Lemmatization*) the following features were built:

- The top 100 words appearing across all songs (excl. second-person pronouns) computed using *Term Frequency (TF)*. 
- The number of times a song appeared on the charts.
- The number of different genres a song belongs to.
- A boolean value indicating whether a song appeared on the *radio airplay chart* of *Billboard* in that period (again only available, in 2023, with Billboard Pro).
- Second Person Pronouns and other word metrics (eg. *cognitive words*, *affect words*, etc) extracted using *Linguistic Inquiry and Word Count* (*LIWC*) 2015.
- 10 different topics extracted by performing *Latent Dirichlet Allocation* (*LDA*) on lyrics (see `LDA.r`).
- A custom version of the *Linguistic Style Matching* (*LSM*) equation, readapted for topic composition in order to measure difference rather than similarity, and computed using the song topic composition and the average topic composition per genre.

For the **Audio Analysis** instead, the following features were built (by far the most technical part, that required some signal processing skills):

- The *Structural Change* of two *Complexity Features* (ie. Chroma and Timbre) as described in Lee et al. (see `Thesis_Notebook.ipynb`). 
- *Arousal* (ibid.).
- *MFCC* (ibid.).

## Analysis

RQ1: Check the statistical significance of the feature engineered variables for the textual analysis was checked using an Ordinary Least Squares linear (OLS) regression, reproducing 10 models described in Berger et al. (see also section 4.3 of my thesis). For the audio analysis instead, 5 models from the Lee et al. paper were reproduced.

RQ2: Three Random Forests and three Support Vector Machines models were built for predicting song’s rank, tuning the hyperparameters using GridSearchCV and selecting the best one in terms of performances (MSE? MAE? che ho usato?)

## Results

Everything is explained thoroughly in my thesis (cf. section 5 and 6).

## (Few) References
- Berger, Jonah and Grant Packard (2020), “Thinking of You: How Second-Person Pronouns Shape Cultural Success”
- Berger, Jonah and Grant Packard (2018), “Are Atypical Things More Popular?”
- Lee, Jong-Seok and Junghyuk Lee (2018), “Music Popularity: Metrics, Characteristics, and Audio-Based Prediction”
