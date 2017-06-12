from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import svm
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

train_file = 'data/twitter-2013train-A.txt'
test_file = 'data/twitter-2013test-A.txt'

def load_data(filename):
	df = pd.read_csv(filename, sep='\t', header=-1, encoding='utf-8')
	df.columns = ['id', 'label', 'tweet']
	df["label"].apply(lambda x: x.strip())
	df['label'] = df['label'].map({'positive': 1, 'negative': -1, 'neutral': 0})
	return df

def fit_score_model(clf, features_train, features_test, df_train, df_test):
	clf.fit(features_train, df_train["label"])
	preds = clf.predict(features_test)

	print clf.best_estimator_

	return classification_report(df_test["label"], preds, digits=3)


with np.load('emb_train.npz') as f:
    emb_train = f['arr_0']

with np.load('emb_test.npz') as f:
    emb_test = f['arr_0']


df_train = load_data(train_file)
df_test = load_data(test_file)

params_linear = [
 	{'C': [1, 10, 100, 1000], 'kernel': ['linear']}
]

params_rbf = [
	{'C': [1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
]

clf_linear = GridSearchCV(estimator=svm.SVC(), param_grid=params_linear, n_jobs=-1, verbose=10)
print "Linear kernel:"
print fit_score_model(clf_linear, emb_train, emb_test, df_train, df_test)

clf_rbf = GridSearchCV(estimator=svm.SVC(), param_grid=params_rbf, n_jobs=-1, verbose=10)
print "RBF kernel:"
print fit_score_model(clf_rbf, emb_train, emb_test, df_train, df_test)
