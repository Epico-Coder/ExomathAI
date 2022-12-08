from pathlib import Path
import pandas as pd
import re

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from cleantext import clean

from sklearn.linear_model import LogisticRegression


def clean_data(df):
    for i in range(df.shape[0]):
        tweet = df['text'][i]
        tweet = clean_text(tweet)
        df['text'][i] = tweet


def clean_text(text):
    emoj = re.compile(
        "["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"u"\U00002702-\U000027B0"u"\U00002702-\U000027B0"u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"u"\U00010000-\U0010ffff"u"\u2640-\u2642"u"\u2600-\u2B55"u"\u200d"u"\u23cf"
        u"\u23e9"u"\u231a"u"\ufe0f"u"\u3030""]+", re.UNICODE)
    text = re.sub(emoj, '', text)

    text = clean(text=text, clean_all=True, extra_spaces=True, stemming=True,
                 stopwords=True, lowercase=True, numbers=True, punct=True,
                 reg="http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                 reg_replace="", stp_lang='english')
    return text


file = (Path(__file__).parent / '../data/media/disaster_tweet.csv').resolve()
df = pd.read_csv(file)
clean_data(df)


X_train, X_test, y_train, y_test = model_selection.train_test_split(
    df["text"],
    df["target"],
    test_size=0.2,
    shuffle=True)

vectorizer = TfidfVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)

lr = LogisticRegression(max_iter=500, solver='liblinear')

# Accuracy: 77-78%
lr_clft = lr.fit(X_train_vector, y_train)

# pickle.dump(lr_clft, open((Path(__file__).parent / '../models/lrclft.pkl').resolve(), 'wb'))
