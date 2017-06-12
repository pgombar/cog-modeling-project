import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

for kernel in ['spectrum', 'subsequence', 'distance']:
	pred = np.asarray(pd.read_csv('out.'+kernel, header=-1)[0])
	gold = np.asarray(pd.read_csv('data/twitter-2013test-A.txt-harry', sep='\t', header=-1)[0])

	print "Kernel: {0}".format(kernel)
	print classification_report(gold, pred, digits=3) + "\n"
