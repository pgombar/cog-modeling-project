import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser()
parser.add_argument('kernels', metavar='k', nargs='+')
args = parser.parse_args()

for kernel in args.kernels:
	pred = np.asarray(pd.read_csv('out.'+kernel, header=-1)[0])
	gold = np.asarray(pd.read_csv('data/twitter-2013test-A.txt-harry', sep='\t', header=-1)[0])

	print "Kernel: {0}".format(kernel)
	print classification_report(gold, pred) + "\n"