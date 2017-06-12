import gensim
import pandas as pd
import numpy as np

train_file = 'data/twitter-2013train-A.txt'
test_file = 'data/twitter-2013test-A.txt'

def load_data(filename):
    df = pd.read_csv(filename, sep='\t', header=-1, encoding='utf-8')
    df.columns = ['id', 'label', 'tweet']
    df["label"].apply(lambda x: x.strip())
    df['label'] = df['label'].map({'positive': 1, 'negative': -1, 'neutral': 0})
    return df

def avg_embedding(words, model, num_features=300):
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.

    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def generate(df, filename, model):
    i = 0
    q = np.zeros((df.shape[0],300), dtype="float32")

    for index, row in df.iterrows():
        text = row['tweet'].lower().split()
        q[i] = avg_embedding(text, model)
        print "{0}: {1}".format(filename, i)
        i += 1

    np.savez_compressed(filename, q)


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print "Loaded w2vec model"

df_train = load_data(train_file)
df_test = load_data(test_file)
print "Loaded data"

generate(df_train, 'emb_train', model)
print "Generated train"
generate(df_test, 'emb_test', model)
print "Generated test"