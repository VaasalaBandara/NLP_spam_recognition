# -*- coding: utf-8 -*-

import pandas as pd #dataframe processing
import nltk #nlp library/popular


df=pd.read_csv("spam.csv",encoding="latin-1")#csv files have different encoding, by default=utf-8

df.head(5)

df.shape

df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True) #inplace=true implies that the resulting dataframe replaces the original one
                                                                    #the columns are specified for drop

df.rename(columns={'v1':'class','v2':'sms'},inplace=True)#replacing the columns v1 and v2 with names,inplace i,plies dataframe replaced

df.sample(5)

df.groupby('class').describe() #group accoring tp class and give statistics

#the unique column of ststistics specify the duplicates. we need to remove duplicates

df=df.drop_duplicates(keep='first')

df.groupby('class').describe()

"""data visualization"""

df['length']=df['sms'].apply(len) #a new column is created with the lengths of sms column called length

df.head(2)

df.hist(column='length',by='class',bins=50)#histogram representing how the length varies, bin size is 50, increases 50 by 50

"""the spam messages seem to have a larger length. this will be learned by the machine

**step2: preprocessing**
"""

from nltk.stem.porter import PorterStemmer #lemetization with porterstemmer/ stem of words are taken

nltk.download('stopwords')
from nltk.corpus import stopwords #download and import stopwords that are available in nltk

nltk.download('punkt')
ps=PorterStemmer()

df.head(5)



"""preprocessing tasks
  1. preprocessing
  2. tokenization
  3. removing special characters
  4. removing stop words and punctuation
  5. stemming
"""

import string

def clean_text(text): #function called clean text is defined with the variable text
  text=text.lower() #lowercase of the sms
  text=nltk.word_tokenize(text) #apply tokenization

  y=[]
  for i in text:
    if i.isalnum():#if text contains only alphanumeric values
      y.append(i) #if not then appended to list

  text=y[:]
  y.clear()

  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation: #checking if it has stop words or punctuations
      y.append(i) #if not then appended to list

  text=y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i)) #the stem of the word is obtained

  return" ".join(y) # text is recreated by joining all tokens with white text

df['sms_cleaned']=df['sms'].apply(clean_text) #for each entry in sms column the clean function is applyed and put in new column sms cleaned

df.head(5)

"""**step 3: feature extraction**"""

from sklearn.feature_extraction.text import TfidfVectorizer #from the sk learn library
                                                            #TfidVectorizer is used

tf_vec=TfidfVectorizer(max_features=3000) #the vectorizor is defined as tf_vec with the vocabulary of 3000/top 3000 words/vector size is vocabulary size or feature size
X=tf_vec.fit_transform(df['sms_cleaned']).toarray() #transforming text to numbers and obtaining the array of it

X.shape #5369 unique sms and 3000 dimensioanl vector/vocabulary size is 3000

y=df['class'].values #the outputs of model

"""**step 4: learning**"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.naive_bayes import MultinomialNB #an algirthm called naive bayes,

model=MultinomialNB()
model.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))

"""which means that 97 percentage of predictions are accurate"""



