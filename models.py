import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

train_file = 'data/twitter-2013train-A.txt'
test_file = 'data/twitter-2013test-A.txt'

def load_data(filename):
	df = pd.read_csv(filename, sep='\t', header=-1, encoding='utf-8')
	df.columns = ['id', 'label', 'tweet']
	df["label"].apply(lambda x: x.strip())
	df['label'] = df['label'].map({'positive': 1, 'negative': -1, 'neutral': 0})
	return df


def fit_score_model(features_train, features_test, df_train, df_test):
	clf = LinearSVC()
	clf.fit(features_train, df_train["label"])
	preds = clf.predict(features_test)

	return classification_report(df_test["label"], preds, digits=3)


df_train = load_data(train_file)
df_test = load_data(test_file)
print "Loaded data"

vectorizer_ngrams = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = "english", \
                             min_df = 3, \
                             ngram_range = (1,3), \
                             max_features = 10000)

vectorizer_chars = CountVectorizer(analyzer = "char",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = "english", \
                             min_df = 3, \
                             ngram_range = (2,5), \
                             max_features = 10000)

features_ngrams_train = vectorizer_ngrams.fit_transform(df_train["tweet"])
features_ngrams_test = vectorizer_ngrams.transform(df_test["tweet"])
features_chars_train = vectorizer_chars.fit_transform(df_train["tweet"])
features_chars_test = vectorizer_chars.transform(df_test["tweet"])
print "Constructed features"

print "Word n-grams:"
print fit_score_model(features_ngrams_train, features_ngrams_test, df_train, df_test)

print "Character n-grams:"
print fit_score_model(features_chars_train, features_chars_test, df_train, df_test)
