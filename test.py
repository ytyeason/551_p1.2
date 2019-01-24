"""
The goal of this module is to process raw json file into inputs X and outputs y 
The important function in process_features:
    takes the proj1_data.json file  as argument
    and returns X, y which are numpy arrays 
    X is 2d 12000 x 163,  y is 1d 12000 x 1
"""

import sys
import numpy as np

def read_json_file(f):
    import os
    import json
    data = []
    with open(os.path.abspath(f)) as fp:
        data = json.load(fp)

    return data

def process_features(data, most_frequent_words):
    
    np.set_printoptions(threshold=np.nan)

    X = []
    y = []

    for row in data:
        y.append(row["popularity_score"])
        x_i = [int(row["children"]), float(row["controversiality"]), int(bool(row["is_root"]))] + process_text(row["text"], most_frequent_words)
        X.append(x_i)

    return np.array(X), np.array(y)

def count_words(text, k):
    all_words = {}
    for string in text:
        string = string.lower()
        words = string.split()
        for word in words:
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1
    
    top_k_words = sorted(all_words.items(), key=lambda t: t[1], reverse=True)[0:k]
    
    r = [tup[0] for tup in top_k_words] 
    print(r)
    return r

def process_text(text, most_frequent_words):
    words = text.lower().split()
    word_counts = {}
    word_features = []

    for word in words:
        if word in word_counts:
            word_counts[word] +=1
        else:
            word_counts[word] = 1

    for word in most_frequent_words:
        if word in word_counts:
            word_features.append(word_counts[word])
        else:
            word_features.append(0)
    # print(word_features)
    return word_features

def train_validate_test_split(X, y):
    train_X, train_y = X[:10], y[:10]
    validate_X, validate_y = X[10000:11000], y[10000:11000]
    test_X, test_y = X[11000:12000], y[11000:12000]

    # print(train_X)

    return train_X, train_y, validate_X, validate_y, test_X, test_y

def w_closed(X,Y):
    dim = np.array(X).shape[1]
    Xarg = np.insert(X,dim,1,axis=1)
    temp1 = np.dot(Xarg.T,Xarg)
    temp2 = np.dot(Xarg.T,Y)
    w = np.dot(np.linalg.inv(temp1),temp2)
    return w

def w_gradient(X,Y,eta=0.5,beta=0.1,e=0.000001):
    dim = np.array(X).shape[1]
    Xarg = np.insert(X,dim,1,axis=1)
    # print(Xarg)

    dim += 1

    weight = np.zeros((dim,1)) #np.array([[0.0],[0.0]])
    alpha = 0.01
    err = np.ones((dim,1)) #np.array([[10.0],[10.0]])
    # i = 1
    while np.linalg.norm(err,2) > e:
        past = weight
        weight = weight - 2*alpha*(Xarg.T @ Xarg @ weight)- Xarg.T @ Y
        err = weight-past
    return weight

def main(args):
    data = read_json_file(args[0])
    most_frequent_words = count_words([row["text"] for row in data], 160)
    
    X, y = process_features(data, most_frequent_words)
    train_X, train_y, validate_X, validate_y, test_X, test_y = train_validate_test_split(X, y)
    # print("INPUT MATRIX X: \n{}".format(X))
    # print("OUTPUT VECTOR Y: \n{}".format(y))
    # print(w_closed(train_X,train_y))
    # print(w_gradient(train_X, train_y))

if __name__ == "__main__": main(sys.argv[1:])