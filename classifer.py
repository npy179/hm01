#!/usr/bin/python
import csv
import numpy as np
from scipy.spatial.distance import euclidean
import heapq
from collections import Counter
import random as rd

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
    predict_test_Y = []
    for i in range(row):
        #distance = dist(test_X[i,:] - )
        distances = [euclidean(test_X[i,:],train_X[j,:])
                     for j in xrange(train_X.shape[0])]
        dist_arr = np.array(distances)
        indices = heapq.nlargest(k, range(len(dist_arr)), dist_arr.take)
        labels_check = [train_Y[i] for i in indices]
        count = Counter(labels_check)
        label, number = count.most_commourn()[0]
        predict_Y.append(label)
    return predict_test_Y

def condensedata(train_X, train_Y,k=1):
    #initialize
    set_index = list(range(train_X.shape[0]))
    condensedIdx = [rd.randint(0, 10000)]
    train_Y_arr = np.array(train_Y)

    while len(condensedIdx) <= len(train_Y):
        rest_predict_label = []
        subset_feature = train_X[condensedIdx,:]
        rest_index = [item for item in set_index if item not in condensedIdx]
        rest_feature = train_X[rest_index,:]
        rest_target = train_Y_arr[rest_index].tolist()
        #find how many case in rest feature
        num_rest_case = len(rest_index)
        for i in range(num_rest_case):
            distances = [euclidean(rest_feature[i,:],subset_feature[j,:])
                         for j in xrange(subset_feature.shape[0])]
            dist_arr = np.array(distances)
            index = heapq.nlargest(k, range(len(dist_arr)), dist_arr.take)
            label = train_Y_arr[index][0]
            rest_predict_label.append(label)

        wrong_predict = [rest_index[i] for i in range(num_rest_case) if
                         rest_target[i] == rest_predict_label[i]]

        if len(wrong_predict) != 0:
            next_index = rd.choice(wrong_predict)
            condensedIdx.append(next_index)
            print condensedIdx
        else:
            break
    return condensedIdx

def main():
    train_X, train_Y, test_X, test_Y = load_cvs("./data/letter-recognition.data.txt")
    condensedIdx = condensedata(train_X,train_Y)
    print condensedIdx
    """
    test_Y_predict = testknn(train_X, train_Y, test_X, 2)
    assert len(test_Y_predict) == len(test_Y), "test_Y_prediction and test_Y do not have equal length"
    match = [1 if test_Y_predict[i]==test_Y[i] else 0 for i in xrange(len(test_Y))]
    match_arr = np.array(match)
    accuracy = match_arr.sum()/match_arr.shape[0]
    print "accuracy is: "+ accuracy
    """


if __name__=="__main__":
    main()
