import re, nltk, os, statistics, random
from gensim.utils import simple_preprocess
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from os.path import expanduser
import pandas as pd
import numpy as np
random.seed(42)

df = pd.read_excel("outdatasetUNO.xlsx")

for x in range(len(df.Lyrics)):
    if df.Lyrics[x]!=0:
        while df.Lyrics[x].find("[")!=-1:
            s = df.Lyrics[x]
            df.Lyrics[x] = s[:s.find("[")] +s[s.find("]")+1:]

for x in range(len(df.Lyrics)):
    df.Lyrics[x] = re.sub(r'\([^)]*\)', '', df.Lyrics[x])

df.Lyrics=df.Lyrics.map(lambda x: re.sub('[\",.\n!?:]', ' ', str(x)))
df.Lyrics=df.Lyrics.map(lambda x: re.sub('    ', ' ', str(x)))
df.Lyrics=df.Lyrics.map(lambda x: re.sub('   ', ' ', str(x)))
df.Lyrics=df.Lyrics.map(lambda x: re.sub('  ', ' ', str(x)))

#2-letter words + weird final characters of lyrics
for x in range(len(df.Lyrics)):
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    df.Lyrics[x] = shortword.sub('', df.Lyrics[x])
    if df.Lyrics[x][-32:] == ' Embed Share Url Copy Embed Copy':
        df.Lyrics[x] = df.Lyrics[x][:-32]

def wordnet_pos_gen(lista):
    pos_tags = list(nltk.pos_tag(lista))
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    t = []
    for x in pos_tags:
        try:
            t.append((x[0], tag_dict[x[1][0]]))
        except Exception:
            t.append(
                (x[0], "n"))  # a get around to assign irrelevant tags to one of wordnet classes, noun in this case
    return t

# Stopping
def stopping(ls, *args):
    processed = []
    for case in ls:
        tok = word_tokenize(case)
        stop_words = set(stopwords.words("english"))#+swToAdd)
        stop_words.add(","), stop_words.add("."), stop_words.add(";"), stop_words.add(":")
        res = [x for x in tok if not x in stop_words]
        result = ""
        for x in res:
            result = result + " " + x
        processed.append(result)
    return processed

# Lemmatizer
def lemma(lista):
    out = []
    t = wordnet_pos_gen(lista)
    lemmatizer = WordNetLemmatizer()
    res = [lemmatizer.lemmatize(w[0], w[1]) for w in t]
    result = ""
    for x in res:
        result = result + " " + x
    out.append(result.lower())  # case normalization step here
    return out

lyric_corpus_tokenized = []
tokenizer = RegexpTokenizer(r'\w+')
for lyric in df.Lyrics:
    tokenized_lyric = tokenizer.tokenize(lyric.lower())
    lyric_corpus_tokenized.append(tokenized_lyric)

for s,song in enumerate(lyric_corpus_tokenized):
    filtered_song = []
    for token in song:
        if len(token) > 2 and not token.isnumeric():
            filtered_song.append(token)
    lyric_corpus_tokenized[s] = filtered_song

for y in range(len(df.Lyrics)):
    print(y)
    #df.Lyrics[y] = [x for x in stopping(lemma(df.Lyrics[y].split(" "))[0].split(" "), swToAdd) if x!=""]
    df.Lyrics[y] = [x for x in stopping(lemma(lyric_corpus_tokenized[y])) if x != ""]

for x in range(len(df.Lyrics)):
    s = ""
    for y in df.Lyrics[x]:
        s+=y
    df.Lyrics[x] = s

df.to_excel("outPostLemma1.xlsx")
