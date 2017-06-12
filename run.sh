#!/bin/sh

harry -c harry.cfg -v -m kern_spectrum $1 spectrum
harry -c harry.cfg -v -m kern_subsequence $1 subsequence
harry -c harry.cfg -v -m kern_distance $1 distance

svm-train -q -v 5 -t 4 spectrum
svm-train -q -v 5 -t 4 subsequence
svm-train -q -v 5 -t 4 distance


# Compute kernel on train set
# Compute kernel on test set
# svm-train makes a model
# svm-predict takes kernel on test set and model file

# 0) generate kernels distance_train and distance_test
# 1) svm-train -t 4 distance_train
# 2) svm-predict distance_test distance.model out.distance > res.distance
