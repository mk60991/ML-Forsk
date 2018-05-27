# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:19:21 2018

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
%config InlineBackend.figure_format = 'retina'

data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()

#Drop column and name change
data = data.drop(["FIELD3: 0", "FILELD4: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

#Drop column and name change
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})

data.tail()

#Count observations in each label
data.label.value_counts()

# convert label to a numerical variable
data['label_num'] = data.label.map({'ham':0, 'spam':1})

data.head()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train) #vect.fit function learns the vocabulary. We can get all the feature names from vect.get_feature_names( ). 

print(vect.get_feature_names()[0:20])
print(vect.get_feature_names()[-20:])

X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)


#visualisation

ham_words = ''
spam_words = ''
spam = data[data.label_num == 1]
ham = data[data.label_num ==0]

import nltk
from nltk.corpus import stopwords

for val in spam.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    #tokens = [word for word in tokens if word not in stopwords.words('english')]
    for words in tokens:
        spam_words = spam_words + words + ' '
        
for val in ham.text:
    text = val.lower()
    tokens = nltk.word_tokenize(text)
    for words in tokens:
        ham_words = ham_words + words + ' '
        
from wordcloud import WordCloud

# Generate a word cloud image
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)

plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
plt.style.use('seaborn-bright')
data.hist(column='length', by='label', bins=50,figsize=(11,5))