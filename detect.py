#---------Imports
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#---------End of imports

### MAIN FLOW ###
if __name__ == '__main__':
    df = pd.read_csv('data/news.csv') # get data
    labels = df.label # labels of fake or real


    x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7) # split dataset

    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7) # initialize a TfidfVectorizer
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) # fit and transform train set,
    tfidf_test=tfidf_vectorizer.transform(x_test) # transform test set

    pac = PassiveAggressiveClassifier(max_iter=50) # initialize a PassiveAggressiveClassifier
    pac.fit(tfidf_train,y_train)

    y_pred = pac.predict(tfidf_test) # predict on the test set
    score = accuracy_score(y_test,y_pred) # calculate accuracy
    print(f'Accuracy: {round(score*100, 2)}%')

    confusion = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL']) # build confusion matrix
    print(confusion)
