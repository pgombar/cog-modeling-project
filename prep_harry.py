import pandas as pd
import sys

file_in = sys.argv[1]
df = pd.read_csv('data/'+file_in, sep='\t', header=-1)

df.columns = ['id', 'label', 'tweet']
df["label"].apply(lambda x: x.strip())
df['label'] = df['label'].map({'positive': 1, 'negative': -1, 'neutral': 0})
del df['id']

df.to_csv('data/'+file_in+'-harry', header=False, index=False, sep='\t')