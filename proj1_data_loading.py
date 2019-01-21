
#             IMPORT            #
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
#################################

import json # we need to use the JSON package to load the data, since the data is stored in JSON format

with open("proj1_data.json") as fp:
    data = json.load(fp)

training_set = data[5:6]
validation_set = data[10000:11000]
testing_set = data[11000:12000]

x_set = []
y_set = []

def pre_process_text(text,frequency_map):
    precessed_list = text.lower().split()

    for word in precessed_list:
            if word in frequency_map:
                frequency_map[word] = frequency_map[word] + 1
            else:
                frequency_map[word] = 1
    return precessed_list

def pre_process(training_set):
    frequency_map = {}
    #preprocess is_root and text
    for item in training_set:
        if item['is_root'] == True:
            item['is_root'] = 1
        else:
            item['is_root'] = 0
        item['text'] = pre_process_text(item['text'],frequency_map)

    #sort the map by key in dscending order
    frequency_map = sorted(frequency_map.items(), key=lambda kv: kv[1], reverse = True)
    #first get the top 160 item of the map, then get the keys into a list
    most_frequent_word = [i[0] for i in frequency_map[0:160]]

    for item in training_set:
        x_counts = [0.0]*160
        for word in item['text']:
            if word in most_frequent_word:
                index = most_frequent_word.index(word)
                x_counts[index] = x_counts[index] + 1.0
        item['w_counts'] = x_counts
        x_set.append(x_counts)#add each data set
        y_set.append([item['popularity_score']])

pre_process(training_set)

X = np.array([[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43],
              [-1.10], [0.40], [-0.96], [0.17]])

X2 = np.array(x_set)

Y = np.array([[2.49], [0.83], [-0.25], [3.10], [0.87], [0.02],
              [-0.12], [1.81], [-0.83], [0.43]])

Y2 = np.array(y_set)

def w_closed(X,Y):
    dim = np.array(X).shape[1]
    Xarg = np.insert(X,dim,1,axis=1)
    temp1 = np.dot(Xarg.T,Xarg)
    temp2 = np.dot(Xarg.T,Y)
    w = inv(temp1).dot(temp2)
    return w

def w_gradient(X,Y,eta=1,beta=1,e=0.0000000001):
    dim = np.array(X).shape[1]
    Xarg = np.insert(X,dim,1,axis=1)
    # print(Xarg)

    dim += 1

    weight = np.zeros((dim,1)) #np.array([[0.0],[0.0]])
    alpha = 0.01
    # e = 0.0000000001
    err = np.ones((dim,1)) #np.array([[10.0],[10.0]])
    # print(weight)
    while err.any() > e:
        past = weight

        print(2*alpha*(np.dot(np.dot(Xarg.T, Xarg), weight)-np.dot(Xarg.T, Y)))

        weight = weight - 2*alpha*(np.dot(np.dot(Xarg.T, Xarg), weight)-np.dot(Xarg.T, Y))

        # print(weight)

        err = abs(weight-past)
    # print(err)
    return weight

# print(w_gradient(X,Y))
print(w_gradient(X2,Y2))
# print(w_closed(X2,Y2))

# print(weight)
# print(Y2[0:10])


#end
