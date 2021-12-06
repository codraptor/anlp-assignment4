import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from collections import Counter

np.random.seed(42)
grid = {
    "n_estimators":np.arange(10,100,10),
    "max_depth":[None,3,5,10],
    "min_samples_split":np.arange(2,20,2),
    "min_samples_leaf":np.arange(1,20,2),
    "max_features": [0.5,1,"sqrt","auto"],
    "max_samples":[10000,6400,15000,6400]
}

X_train = pd.read_csv('./data/train_texts.csv', header=0, sep=',', quotechar='"')
Y_train = pd.read_csv('./data/train_labels.csv', header=0, sep=',', quotechar='"')
X_test = pd.read_csv('./data/test_texts.csv', header=0, sep=',', quotechar='"')
Y_test = pd.read_csv('./data/test_labels.csv', header=0, sep=',', quotechar='"')
X_train = X_train.head(8000)
Y_train = Y_train.head(8000)
X_test = X_test.head(2000)
Y_test = Y_test.head(2000)

def clean(text):
    wn = nltk.WordNetLemmatizer()
    stopword = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    lower = [word.lower() for word in tokens]
    no_stopwords = [word for word in lower if word not in stopword]
    no_alpha = [word for word in no_stopwords if word.isalpha()]
    lemm_text = [wn.lemmatize(word) for word in no_alpha]
    clean_text = lemm_text
    counter = Counter(clean_text)
    most_occur = counter.most_common(40)
    most_occuring_words = [a_tuple[0] for a_tuple in most_occur]
    clean_text = [i for i in clean_text if i in most_occuring_words]
    return clean_text
    
def vectorize(data,tfidf_vect_fit):
    X_tfidf = tfidf_vect_fit.transform(data)
    words = tfidf_vect_fit.get_feature_names()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return(X_tfidf_df)
    
tfidf_vect = TfidfVectorizer(analyzer=clean)
tfidf_vect_fit=tfidf_vect.fit(X_train['train_texts'])
X_train=vectorize(X_train['train_texts'],tfidf_vect_fit)
X_train.head()

X_test=vectorize(X_test['test_texts'],tfidf_vect_fit)
X_test.head()

model = RandomizedSearchCV(
RandomForestRegressor(n_jobs=-1,
                     random_state=42),
                    param_distributions = grid,
                     n_iter=5,
                    cv=5,
                    verbose=True)
                    
model.fit(X_train,Y_train.values.ravel())

y_preds = model.predict(X_test)
mae_hyp = mean_absolute_error(Y_test,y_preds)
print(mae_hyp)


