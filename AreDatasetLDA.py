#ARE - LDA implementation
import re
import gensim
from gensim.utils import simple_preprocess
import nltk
import pandas as pd
import numpy as np
import os
import statistics
import random
random.seed(42)
os.chdir(r"C:\Users\alema\Desktop")

#'''
#df = pd.read_excel("outdataset1.xlsx")
df = pd.read_excel("outdatasetUNO.xlsx")
##cols= list(df.columns.values)[::-1]
##df=df[cols]
#df.Lyrics = df.Lyrics.astype(str)
#df = df.sort_values(by="uniq_id")
#df = df.drop(columns="Unnamed: 0")
#df = df.reset_index()
#df = df.drop(columns="index")
##df.to_excel("outdatasetUNO.xlsx")
#alla riga 1122, quella di "Lee Brice_I Don't Dance", IL WC dice 205, la song originale ne ha 215, hanno tolto 10 parole, cioe nulla!!!!!!
#data cleaning
for x in range(len(df.Lyrics)):
    if df.Lyrics[x]!=0:
        while df.Lyrics[x].find("[")!=-1:
            s = df.Lyrics[x]
            df.Lyrics[x] = s[:s.find("[")] +s[s.find("]")+1:]

for x in range(len(df.Lyrics)):
    df.Lyrics[x] = re.sub(r'\([^)]*\)', '', df.Lyrics[x])

df.Lyrics=df.Lyrics.map(lambda x: re.sub('[\",.\n!?:]', ' ', str(x)))
#df.Lyrics=df.Lyrics.map(lambda x: str(x).lower())
df.Lyrics=df.Lyrics.map(lambda x: re.sub('    ', ' ', str(x)))
df.Lyrics=df.Lyrics.map(lambda x: re.sub('   ', ' ', str(x)))
df.Lyrics=df.Lyrics.map(lambda x: re.sub('  ', ' ', str(x)))

#2-letter words + togli caratteri finali erronei dei lyrics
for x in range(len(df.Lyrics)):
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    df.Lyrics[x] = shortword.sub('', df.Lyrics[x])

    if df.Lyrics[x][-32:] == ' Embed Share Url Copy Embed Copy':
        df.Lyrics[x] = df.Lyrics[x][:-32]




import nltk
from nltk.corpus import wordnet
from os.path import expanduser

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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#swToAdd = []
#with open("stopwords.txt") as f: #addo cose alla lista delle stopwords, così...
#    for x in f:
#        swToAdd.append(x[:-1])
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
from nltk.stem import WordNetLemmatizer
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

from nltk.tokenize import RegexpTokenizer
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





#df.to_excel("outPostLemma1.xlsx")
df = pd.read_excel("outPostLemma1.xlsx")





from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]

data = df.Lyrics.values.tolist()
data_words = list(sent_to_words(data))
# remove stop words
data_words = remove_stopwords(data_words)
print(data_words[:1][0][:30])

import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1][0][:30])

from pprint import pprint
num_topics = 10
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics, random_state=42
                                      )
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

import pyLDAvis.gensim_models
import pickle
import pyLDAvis
pyLDAvis.enable_notebook()
LDAvis_data_filepath = os.path.join(r"C:\Users\alema\Desktop\Thesis\ldavis_prepared_"+str(num_topics))
if 1 == 1:
    LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
with open(LDAvis_data_filepath, 'rb') as f:
    LDAvis_prepared = pickle.load(f)
pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath+ str(num_topics) +'.html')
LDAvis_prepared

'''

####################################################### Other LDA attempts
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

data_samples = df.Lyrics#.iloc[:n_samples]

# Use tf-idf features for NMF.
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data_samples)

# Use tf (raw term count) features for LDA.
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(data_samples)

# Fit the NMF model
nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
plot_top_words(nmf, tfidf_feature_names, n_top_words,
               'Topics in NMF model (Frobenius norm)')

# Fit the NMF model
nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)

tfidf_feature_names = tfidf_vectorizer.get_feature_names()
plot_top_words(nmf, tfidf_feature_names, n_top_words,
               'Topics in NMF model (generalized Kullback-Leibler divergence)')

lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(tf)
tf_feature_names = tf_vectorizer.get_feature_names()
plot_top_words(lda, tf_feature_names, n_top_words, 'Topics in LDA model')



##### ????
# from sklearn.decomposition import NMF
#
# nmf = NMF(n_components=2, random_state=43,  alpha=0.1, l1_ratio=0.5)
# nmf_output = nmf.fit_transform(tfidf_feature_matrix)
#
# nmf_feature_names = tfidf_vectorizer.get_feature_names()
# nmf_weights = nmf.components_


###########
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
# use tfidf by removing tokens that don't appear in at least 50 documents
vect = TfidfVectorizer(min_df=50, stop_words='english')

# Fit and transform
X = vect.fit_transform(data_samples)
model = NMF(n_components=10, random_state=5)

# Fit the model to TF-IDF
model.fit(X)

# Transform the TF-IDF: nmf_features
nmf_features = model.transform(X)
components_df = pd.DataFrame(model.components_, columns=vect.get_feature_names())
components_df

for topic in range(components_df.shape[0]):
    tmp = components_df.iloc[topic]
    print(f'For topic {topic+1} the words with the highest value are:')
    print(tmp.nlargest(10))
    print('\n')

pd.DataFrame(nmf_features).loc[55]
##################

#topic comp of tracks
songTopicComp = []
for i in range(len(df.genre)):
    top_topics = lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(10)]
    songTopicComp.append(topic_vec)

songTopicComp = []
for x in range(len(df.uniq_id)): # QUESTO MI E SERVITO CO DF23 E SOLO CO QUELLO
    songTopicComp.append(list(df[["lda10topic{}".format(x) for x in range(1,11)]].iloc[x]))


#genre’s mean topic composition
genres = ["Dan", "Pop", "Rap", "RnB", "Roc", "Chr", "Cou"]
dic_genres = {"Dan":0, "Pop":1, "Rap":2, "RnB":3, "Roc":4, "Chr":5, "Cou":6}
avgTopComp = []

for x in genres: #NON STO CONSIDERANDO QUELLI CON PIù DI UN GENERE (nel dataset, nel campo a 3 lettere, non sembrano essereci)
    aa = [songTopicComp[y] for y in range(len(df.genre)) if df.genre[y]==x]
    temp=[]
    for y in range(10):
        temp.append(statistics.mean([aa[z][y] for z in range(len(aa))]))
    avgTopComp.append(temp)

#Lyrical differentiation of songs from their genre
#WHAT IF A SONG HAS TWO GENRES?!"?!?!?!?!?
songLyriciff = []
for x in range(len(df.genre)):
    songLyriciff.append(list(abs(np.array(songTopicComp[x]) -
                            np.array(avgTopComp[dic_genres[df.genre[x]]]))))


#Invented Rank Generator!!!!!!!!!!!! RICORDATI CHE è FINTO :] MA IN REALTà NOOOOOOOOOOO, DEVI SOLO SORTARLE
#df = df.sort_values(by="uniq_id")

df["InventedRank"] = 0
i,j = 0, [x for x in range(50,0,-1)] #reversec0ded
while i<=len(df.genre):
    try:
        df.iloc[i:i+50]["InventedRank"] = j
        i+=50
    except ValueError:
        print("all handled :)")
        try:
            if df.InventedRank[i]==0: #useful only for temp_dataset, then will be multiple of 50
                j=1
                for x in range(i,len(df.genre)):
                   df.InventedRank[x] = j
                   j+=1
        except:
            print("mh ok")
            break
        break


#pd.DataFrame(lda.transform(tf)).loc[55] #single row topic comp credo hehe
songTopicComp2 = []
dfTop = pd.DataFrame(lda.transform(tf))
for i in range(len(df.genre)):
    top_topics = dfTop.loc[i]
    topic_vec = [top_topics[i] for i in range(10)]
    songTopicComp2.append(topic_vec)
#genre’s mean topic composition
genres = ["Dan", "Pop", "Rap", "RnB", "Roc", "Chr", "Cou"]
dic_genres = {"Dan":0, "Pop":1, "Rap":2, "RnB":3, "Roc":4, "Chr":5, "Cou":6}
avgTopComp2 = []

for x in genres: #NON STO CONSIDERANDO QUELLI CON PIù DI UN GENERE (nel dataset, nel campo a 3 lettere, non sembrano essereci)
    aa = [songTopicComp2[y] for y in range(len(df.genre)) if df.genre[y]==x]
    temp=[]
    for y in range(10):
        temp.append(statistics.mean([aa[z][y] for z in range(len(aa))]))
    avgTopComp2.append(temp)

#Lyrical differentiation of songs from their genre
#WHAT IF A SONG HAS TWO GENRES?!"?!?!?!?!?
songLyriciff2 = []
songLSMlikediff2 = []
for x in range(len(df.genre)):
    #songLyriciff2.append(list(abs(np.array(songTopicComp2[x]) - np.array(avgTopComp2[dic_genres[df.genre[x]]]))))
    songLSMlikediff2.append(
        statistics.mean(list(np.divide(abs(np.array(songTopicComp[x]) - np.array(avgTopComp[dic_genres[df.genre[x]]])),
                           np.array(songTopicComp[x]) + np.array(avgTopComp[dic_genres[df.genre[x]]]) + 0.0001)))
    )

######### LStyleM per genre
ao = pd.read_excel("LIWC2015 Results (soloLyrics2).xlsx")
ao = ao.sort_values(by="Source (A)")
categ = ["ppron", "ipron", "article", "auxverb", "adverb", "prep", "conj", "quant", "negate"]
avgStylePerGen = []
for q in genres:
        ind = [x for x in range(len(df.genre)) if df.genre[x] == q]
        avgStylePerGen.append([statistics.mean(ao.iloc[ind][x]) for x in categ])

df["lsm_score"] = 0.0
for x in range(len(ao.prep)):
    df["lsm_score"][x] =  statistics.mean(list(np.divide((abs(np.array(avgStylePerGen[dic_genres[df.genre[x]]])) - np.array(ao[categ].iloc[x])) ,
                                  (np.array(avgStylePerGen[dic_genres[df.genre[x]]]) + np.array(ao[categ].iloc[x]) + 0.0001)))) # "1 -" not needed for meg


















# RF on song popularity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error #qui sotto tolgo lda10topic perchè è la loro analisi che avrei rifatto co LDA in teoria
df1 = df[[x for x in list(df.drop(columns=["song_case","month","year","date","Unnamed: 0","uniq_id"] + ["lda10topic"+str(x) for x in range(1,11)]).columns) if not isinstance(df[x][0], str)]]
df1[["myLdaTopics"+str(x) for x in range(1,11)]] = songTopicComp
df1[["songLyricDiff"+str(x) for x in range(1,11)]] = songLyriciff
#df1[["myLdaTopics2"+str(x) for x in range(1,11)]] = songTopicComp2
#df1[["songLyricDiff2"+str(x) for x in range(1,11)]] = songLyriciff2
df1["songLSMlikediff"] = songLSMlikediff

df1 = df1.fillna(0)
'''



# L O L L O L L O L L O L L O L O L



#df1.to_excel("df1LOL.xlsx")
df1 = pd.read_excel("df1LOL.xlsx")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns=["InventedRank"]), df1["InventedRank"], test_size=0.33, random_state=42)

# rf = RandomForestRegressor(random_state = 42)
# rf.fit(X_train,y_train)
# predictions = rf.predict(X_test)
# errors = abs(predictions - y_test)
# mean_squared_error(y_test, predictions, squared=True)
# mean_absolute_error(y_test, predictions)


##[x for x in list(df.drop(columns=["date","Unnamed: 0","uniq_id","song_case"]).columns) if not isinstance(df[x][0], str)]
'''
# Grid for GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_rf = {'n_estimators': [200,1000],
               'max_depth': [10, 100],
               'min_samples_split':[2, 10],
               'min_samples_leaf': [1, 4],
            }

#GridSearch and Model Fit
#something_rf = GridSearchCV(rf, param_grid=grid_rf, cv=5, verbose=2)
from matplotlib import pyplot as plt
something_rf = RandomForestRegressor(random_state=42, max_depth=10, min_samples_leaf=1, min_samples_split=2, n_estimators=1000)
something_rf.fit(X_train, y_train)
something_rf.feature_importances_
something_rf
plt.barh(, something_rf.feature_importances_)
#something_rf.best_params_
predictions = something_rf.predict(X_test)
errors = abs(predictions - y_test)
mean_absolute_error(y_test, predictions)
mean_squared_error(y_test, predictions, squared=True)

## LIME
import lime
import lime.lime_tabular
import webbrowser
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.iloc[:].values, feature_names=df1.columns,mode='regression')
i = 407
exp = explainer.explain_instance(X_test.iloc[i], something_rf.predict, num_features=15)
exp.show_in_notebook(show_table=True)
exp.save_to_file(r'C:\Users\alema\Desktop\Thesis\ecchilo.html')
url = r'C:\Users\alema\Desktop\Thesis\ecchilo.html'
webbrowser.open(url,new=2)   #HTML Page
fig = exp.as_pyplot_figure() #Actual Plot

#[x for x in range(len(X_test.d2014)) if X_test.songLyricDiff1.iloc[x] == max(X_test.songLyricDiff1)]


from sklearn import linear_model
clf = linear_model.PoissonRegressor()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
clf.coef_

from random import seed
seed(42)
from sklearn import linear_model
reg = linear_model.LinearRegression()
X_train.drop([])
reg.fit(X_train, y_train)
sorted([(reg.coef_[x], X_train.columns[x]) for x in range(len(reg.coef_))])
predictions = reg.predict(X_test)
errors = abs(predictions - y_test)
mean_absolute_error(y_test, predictions)
'''










from scipy import stats

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X_train.drop(columns=["songLyricDiff{}".format(x) for x in range(1,11)]).columns
vif_data["VIF"] = [variance_inflation_factor(X_train.drop(columns=["songLyricDiff{}".format(x) for x in range(1,11)]).values, i)
                   for i in range(len(X_train.drop(columns=["songLyricDiff{}".format(x) for x in range(1,11)]).columns))]
print(vif_data)

# vif_data = pd.DataFrame()
# vif_data["feature"] = X_train[["songLSMlikediff", "d2015"]].columns
# vif_data["VIF"] = [variance_inflation_factor(X_train[["songLSMlikediff", "d2015"]].values, i)
#                    for i in range(len(X_train[["songLSMlikediff", "d2015"]].columns))]
# print(vif_data)


#PCA NON SI APPLICA ALLE VARIABILI DISCRETE O CATEGORICHE, SOLO CONTINUE ;D

import matplotlib as plt
import seaborn as sns

sns.heatmap(X_train[my_cols].corr(), annot=True)
'''
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(df1.drop(columns = ["InventedRank"]))
scaled_features_df = pd.DataFrame(scaled_features, index=df1.index, columns=df1.drop(columns = ["InventedRank"]).columns)
scaled_features_df = scaled_features_df.fillna(0)
scaled_features = scaled_features_df.to_numpy()
from sklearn.decomposition import PCA
############# #components for PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(scaled_features_df)
reduced = pca.transform(scaled_features_df)
principalDf = pd.DataFrame(data = reduced , columns = ['principal component {}'.format(x) for x in range(1,len(reduced[0])+1)]) #really 116?!?!?
sns.heatmap(principalDf.corr(), annot=True)

## Funge, ti plotta un grafichetto carino
# import matplotlib.pyplot as plt
# pca = PCA().fit(scaled_features)
# plt.rcParams["figure.figsize"] = (12,6)
#
# fig, ax = plt.subplots()
# xi = np.arange(1, 143, step=1)
# y = np.cumsum(pca.explained_variance_ratio_)
#
# plt.ylim(0.0,1.1)
# plt.plot(xi, y, marker='o', linestyle='--', color='b')
#
# plt.xlabel('Number of Components')
# plt.xticks(np.arange(1, 143, step=1)) #change from 0-based array index to 1-based human-readable label
# plt.ylabel('Cumulative variance (%)')
# plt.title('The number of components needed to explain variance')
#
# plt.axhline(y=0.95, color='r', linestyle='-')
# plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)
#
# ax.grid(axis='x')
# plt.show()




########
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(scaled_features_df)
principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6', 'principal component 7', 'principal component 8', 'principal component 9', 'principal component 10'])
sns.heatmap(principalDf.corr(), annot=True)
#'''


date_dic = {'q1_14':1, 'q2_14':2,'q3_14':3, 'q4_14':4, 'q1_15':5, 'q2_15':6, 'q3_15':7, 'q4_15':8, 'q1_16':9, 'q2_16':10, 'q3_16':11, 'q4_16':12}
#le dummy dei generi so correlate al 100 percento coddue
genres = ["Dan", "Pop", "Rap", "RnB", "Roc", "Chr", "Cou"]
dic_genres = {"Dan":0, "Pop":1, "Rap":2, "RnB":3, "Roc":4, "Chr":5, "Cou":6}
#df = pd.read_excel("outdataset1.xlsx")
#df=0
df1["genre"] = [dic_genres[df.genre[x]] for x in range(len(df1.d2014))]
df1 = df1.drop(columns = ["g_{}".format(x.lower()) for x in genres])
df1["date"] = 0
for x in list(date_dic.keys()):
    for y in range(len(df1.genre)):
        if df1[x][y] == 1:
            df1["date"][y] = date_dic[x]
df1 = df1.drop(columns = ["d2014","d2015","d2016",'q1_14', 'q2_14','q3_14', 'q4_14', 'q1_15', 'q2_15', 'q3_15', 'q4_15', 'q1_16', 'q2_16', 'q3_16', 'q4_16'])



X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns=["InventedRank"]), df1["InventedRank"], test_size=0.33, random_state=42)


import re
df["artist1"] = 0
a = list(df.artist)
processed=[]
exclusions = '|'.join([" with ", " featuring ", " feat ", " & ", ", ", " ft. ", " ft ", " and ", " \+ ", " x "])
for x in a:
    x = re.sub(exclusions, "|", x.lower())
    x = re.sub("  ", "", x)
    processed.append(x.split("|"))

#selfDummyCreator
totalCols = []
for x in processed:
    print(x)
    for y in x:
        totalCols.append(y)
totalCols = set(totalCols)
for x in totalCols:
    print(x)
    df1[x] = 0

for x in range(len(df1.artist)):
    for y in processed[x]:
        df1[y][x] = 1

df1[list(df1.columns)[:144]]

my_cols =  ["d_airplaysong", "count_genres", "WC", "songLSMlikediff", "genre", "date"]
my_cols2= list(df1.drop("songLyricDiff1", "songLyricDiff2", "songLyricDiff3","songLyricDiff4", "songLyricDiff5", "songLyricDiff6", "songLyricDiff7", "songLyricDiff8",
                        "songLyricDiff9", "songLyricDiff10"

).columns)

# [ "myLdaTopics1", "myLdaTopics2", "myLdaTopics3", "myLdaTopics4", "myLdaTopics5", "myLdaTopics6", "myLdaTopics7",
#       "myLdaTopics8", "myLdaTopics9", "myLdaTopics10", "songLyricDiff1", "songLyricDiff2", "songLyricDiff3",
#       "songLyricDiff4", "songLyricDiff5", "songLyricDiff6", "songLyricDiff7", "songLyricDiff8", "songLyricDiff9",
#       "songLyricDiff10","d_airplaysong", "count_genres", "WC", "songLSMlikediff", "genre"]



import statsmodels.api as sm
#data_exog = sm.add_constant(X_train[my_cols])
data_exog = sm.add_constant(X_train.drop(columns=["songLyricDiff{}".format(x) for x in range(1,11)]))
#data_exog = sm.add_constant(X_train)
res = sm.OLS(y_train, data_exog).fit()
print(res.summary())
ale=pd.DataFrame(res.pvalues)

#
scaled_features_df["InventedRank"] = df1.InventedRank
X_train, X_test, y_train, y_test = train_test_split(scaled_features_df.drop(columns=["InventedRank"]), df1["InventedRank"], test_size=0.33, random_state=42)
data_exog = sm.add_constant(X_train)
res = sm.OLS(y_train, data_exog).fit()
print(res.summary())
ale=pd.DataFrame(res.pvalues)















'''
from sklearn import linear_model
from regressors import stats
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)

# To calculate the p-values of beta coefficients:
print("coef_pval:\n", stats.coef_pval(ols, X_train, y_train))

# to print summary table:
print("\n=========== SUMMARY ===========")
xlabels = list(df1.drop(columns=["InventedRank"]).columns)
stats.summary(ols, X, y, xlabels)



#
#############
# ###Language Style Matchin Equation
# categ = ["ppron", "ipron", "article", "auxverb", "adverb", "prep", "conj", "quant", "negate"]
#
# #df.Lyrics.to_excel("soloLyrics2.xlsx")
# ao = pd.read_excel("LIWC2015 Results (soloLyrics2).xlsx")
# avg_style = [statistics.mean(ao[x]) for x in cat] #mi tocca usà la mia analisi perché loro non hanno 3 4 classi su 9 mcxvcxbnbvmxc
# for y in range(len(categ)):
#     lsm_score = 1 - [(abs(avg_style[y] - ao[categ[y]][x] )) for x in range(len(ao.prep))] # per genere
'''