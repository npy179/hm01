#!/usr/bin/python
import csv
import numpy as np
from scipy.spatial.distance import euclidean
import heapq
from collections import Counter

def load_cvs(dataset_name):
    labels = []
    features = []
    with open(dataset_name,"r") as cvsfile:
        csv_content = csv.reader(cvsfile,delimiter=",",quotechar="'")
        for row in csv_content:
            features.append(map(int,row[1:]))
            labels.append(row[0])

        features_matrix = np.vstack(features)

    train_X = features_matrix[:15000]
    train_Y = labels[:15000]

    test_X = features_matrix[15000:]
    test_Y = labels[15000:]
    return train_X, train_Y, test_X, test_Y

def testknn(train_X, train_Y, test_X, k):
    #return test_Y
    row, col = test_X.shape
    test_Y = []
    for i in range(row):
        #distance = dist(test_X[i,:] - )
        distances = [euclidean(test_X[i,:],train_X[j,:]) for j in xrange(train_X.shape[0])]
        dist_arr = np.array(distances)
        indices = heapq.nlargest(k, range(len(dist_arr)), dist_arr.take)
        labels_check = [train_Y[i] for i in indices]
        count = Counter(labels_check)
        label, number = count.most_common()[0]
        test_Y.append(label)
    return test_Y


def main():
    train_X, train_Y, test_X, test_Y = load_cvs("./data/letter-recognition.data.txt")
    test_Y = testknn(train_X, train_Y, test_X, 2)
    for label in test_Y:
        print "label: "+label
if __name__=="__main__":
    main()
