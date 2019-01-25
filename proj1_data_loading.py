#             IMPORT            #
import matplotlib.pyplot as plt
import numpy as np
import json # we need to use the JSON package to load the data, since the data is stored in JSON format

# np.set_printoptions(threshold=np.nan)
#################################

# task 1

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
training_set = data[0:10000]
validation_set = data[10000:11000]
testing_set = data[11000:12000]

def pre_process_text(text,frequency_map):
    precessed_list = text.lower().split()

    for word in precessed_list:
            if word in frequency_map:
                frequency_map[word] = frequency_map[word] + 1
            else:
                frequency_map[word] = 1
    return precessed_list

def pre_process(data_set):
    frequency_map = {}
    #preprocess is_root and text
    for item in data_set:
        if item['is_root'] == True:
            item['is_root'] = 1
        else:
            item['is_root'] = 0
        item['text'] = pre_process_text(item['text'],frequency_map)

    #sort the map by key in dscending order
    frequency_map = sorted(frequency_map.items(), key=lambda kv: kv[1], reverse = True)
    #print(frequency_map[0:160])
    #first get the top 160 item of the map, then get the keys into a list
    most_frequent_word = [i[0] for i in frequency_map[0:160]]
    # print(most_frequent_word)
    for item in data_set:
        x_counts = [0.0]*160
        for word in item['text']:
            if word in most_frequent_word:
                index = most_frequent_word.index(word)
                x_counts[index] = x_counts[index] + 1.0
        item['w_counts'] = x_counts

    return data_set

def getXandY(data):
    x_set = []
    y_set = []
    for item in data:
        x_others = [1]#, np.log(item['children'] + 1), np.power(item['children'],2)] # 2 other features + bias
        x_set.append(x_others + item['w_counts'])  #add each data set
        y_set.append([item['popularity_score']])
    return np.array(x_set), np.array(y_set)

# get x and y
training_set = pre_process(training_set)
x_train, y_train = getXandY(training_set)

validation_set = pre_process(validation_set)
x_valid, y_valid = getXandY(validation_set)

# test data --------
X = np.array([[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43],
              [-1.10], [0.40], [-0.96], [0.17]])

Y = np.array([[2.49], [0.83], [-0.25], [3.10], [0.87], [0.02],
              [-0.12], [1.81], [-0.83], [0.43]])

# ------------------

# task 2
def w_closed(X,Y):
#    dim = np.array(X).shape[1]
#    Xarg = np.insert(X,dim,1,axis=1) # add bias
    
    temp1 = np.dot(X.T,X)
    temp2 = np.dot(X.T,Y)
    
    w = np.dot(np.linalg.inv(temp1),temp2)
    return w

# epsilon should be <= 10^-6, eta < 10^-5, beta < 0.001
def w_gradient(X,Y,eta=10e-07,beta=0.0009,e=10e-5):
    dim = np.array(X).shape[1]
#    Xarg = np.insert(X,dim,1,axis=1) # include bias
    # Xarg = np.c_[X, [1]*1000]

    weight = np.random.random((dim,1))
#    print(weight)
    # print(-2*(np.dot(np.dot(Xarg.T, Xarg), weight) - np.dot(Xarg.T, Y)))
    # print(np.dot(Xarg.T, Y))

    diff = np.ones((dim,1))

    i = 1
    while np.linalg.norm(diff,2) > e:
        alpha = eta / (1 + beta * i)
        # alpha = 0.01
        # past = weight.copy()

        # print(2*alpha*(np.dot(np.dot(Xarg.T, Xarg), weight)-np.dot(Xarg.T, Y)))
        diff = 2*alpha*(np.dot(np.dot(X.T, X), weight)-np.dot(X.T, Y))
        weight = weight - diff
        #print(np.linalg.norm(diff,2))
        
        # err = 2*alpha*(np.dot(np.dot(Xarg.T, Xarg), weight)-np.dot(Xarg.T, Y))
#        print(weight)
#        print("   ",alpha)
        print(np.linalg.norm(diff,2))

#         print(np.linalg.norm(err,2))

        i+=1
    print(i)
    return weight

def MSE(X,w,y):
    xw = X @ w
    return (y - xw).T @ (y-xw) / len(y)

#print(w_closed(X,Y))
#print(w_gradient(X,Y))

# w_gradient(X2,Y2)

print ("-----------------------------------------------------------------")

#weight_c = w_closed(x_train,y_train)
#
#print("weight in closed form is: \n", weight_c)
#print("MSE for closed form is: \n", MSE(x_train, weight_c, y_train))

weight_g = w_gradient(x_train,y_train)
print("weight gradient is: \n", weight_g)
print("MSE for gradient is: \n", MSE(x_train, weight_g, y_train))







#end
