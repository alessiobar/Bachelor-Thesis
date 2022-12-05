# Song-Popularity-Prediction
This repository contains the code of my Bachelor's thesis, titled "*Prediction of song popularity with a textual and audio based approach*", and available for download at http://tesi.luiss.it/31943/ (alternatively see `Thesis.pdf`).

This thesis proposes a blended approach for predicting song popularity combining some textual based approaches presented in Berger et al. (2018; 2020), with an audio based one suggested in Lee et al. (2018).

## Data

For the **text mining** part, the goal was to rebuild from scratch the original dataset used by Berger et al., which contains song-related information about 4200 songs and it was scraped originally from *Billboard.com* (see *You_S1_Data_NoBillboardRanking.csv* at https://osf.io/cbguq).

Since two different types of analyses are carried out, ie. Text Mining and Audio Mining, two datasets will be used: 
- The first contains song-related information about 4200 songs, it was made available by Berger et al. (see *You_S1_Data_NoBillboardRanking.csv* at https://osf.io/cbguq), and it was scraped originally from *Billboard.com*. However, the lyrics were not present and were added using `AddLyrics.py`. (Refer instead to `BillboardScraper.py` to acquire any new/different data; however notice that, as of now (2022), the *Digital Song Sales charts* are accessible only by paying a monthly fee).
- The other one contains the corresponding audio tracks, downloaded in mp3 format from *Youtube.com* (see `YoutubeToMp3.py`)

## Feature Engineering
Most of the variables needed by the models had to be feature engineered, especially the ones for audio data.

Textual Analysis Features:
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


PS se non voglio che si vedano i commit, posso crearne diretto un altra di repo, questa la lascio privata al limite o la cancello
