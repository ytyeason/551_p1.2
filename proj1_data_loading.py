#             IMPORT            #
import matplotlib.pyplot as plt
import numpy as np
import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from datetime import datetime 

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

def pre_process(data_set, nb_of_top_words=160):
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
    most_frequent_word = [i[0] for i in frequency_map[0:nb_of_top_words]]
    # print(most_frequent_word)
    for item in data_set:
        x_counts = [0.0]*nb_of_top_words
        for word in item['text']:
            if word in most_frequent_word:
                index = most_frequent_word.index(word)
                x_counts[index] = x_counts[index] + 1.0
        item['w_counts'] = x_counts

    return data_set

def getXandY(data,with_text_feature=1):
    x_set = []
    y_set = []
    for item in data:
        if (with_text_feature == 1):
            x_others = [1, np.log(item['children'] + 1), item['controversiality']]#, np.power(item['children'],2)] # 2 other features + bias
            x_set.append(x_others + item['w_counts'])  #add each data set
        else:
            x_set.append([1, item['children'], item['controversiality'], item['is_root']])
        y_set.append([item['popularity_score']])
    return np.array(x_set), np.array(y_set)

# task 2
def w_closed(X,Y):
#    dim = np.array(X).shape[1]
#    Xarg = np.insert(X,dim,1,axis=1) # add bias
    
    temp1 = np.dot(X.T,X)
    temp2 = np.dot(X.T,Y)
    
    w = np.dot(np.linalg.inv(temp1),temp2)
    return w

# epsilon should be <= 10^-6, eta < 10^-5, beta < 0.001
def w_gradient(X,Y,eta=1e-06,beta=1e-05,e=1e-6):
    dim = np.array(X).shape[1]
#    Xarg = np.insert(X,dim,1,axis=1) # include bias

    weight = np.random.random((dim,1))
    diff = np.ones((dim,1))
    
    # for faster computation
    xtx = np.dot(X.T, X)
    xty = np.dot(X.T, Y)
    
#    # test convergence
#    past_diff = 10
    
    i = 1
    while np.linalg.norm(diff,2) > e:
        alpha = eta / (1 + beta * i)

        # print(2*alpha*(np.dot(np.dot(Xarg.T, Xarg), weight)-np.dot(Xarg.T, Y)))
        diff = 2*alpha*(np.dot(xtx, weight)-xty)
        weight = weight - diff
        #print(np.linalg.norm(diff,2))
        
#        # test convergence
#        if (i==1):
#            past_diff = np.linalg.norm(diff,2)
##            print(np.linalg.norm(diff,2))
#        if (i == 2 and past_diff <= np.linalg.norm(diff,2)):
##            print(np.linalg.norm(diff,2))
#            print("in 2nd if")
#            return np.ones((dim,1))
##         print(np.linalg.norm(err,2))

        i+=1
#    print(i)
    return weight

def MSE(X,w,y):
    xw = X @ w
    return ((y - xw).T @ (y-xw) / len(y))[0][0]

#print(w_closed(X,Y))
#print(w_gradient(X,Y))

# w_gradient(X2,Y2)

#print ("-----------------------------------------------------------------")
#
#weight_c = w_closed(x_train,y_train)
#
#print("weight in closed form is: \n", weight_c)
#print("training MSE for closed form is: \n", MSE(x_train, weight_c, y_train))
#print("validating MSE for closed form is: \n", MSE(x_valid, weight_c, y_valid))
#
#weight_g = w_gradient(x_train,y_train)
#print("weight gradient is: \n", weight_g)
#print("training MSE for gradient is: \n", MSE(x_train, weight_g, y_train))
#print("validating MSE for gradient is: \n", MSE(x_valid, weight_g, y_valid))

# task 3
print ("TASK 3 -----------------------------------------------------------------")
training_set = pre_process(training_set,60)
validation_set = pre_process(validation_set,60)

print("Testing with 3 simple features")
# get x and y
x_train, y_train = getXandY(training_set,0)
x_valid, y_valid = getXandY(validation_set,0)

startTime = datetime.now()
w_closed_train_3Simple = w_closed(x_train,y_train)
endTime = datetime.now()
print("Closed form took: ", endTime-startTime, "minutes")

startTime = datetime.now()
w_gd_train_3Simple = w_gradient(x_train,y_train)
endTime = datetime.now()
print("Gradient took: ", endTime-startTime, "minutes")






'''
# to test which values to give to hyperparameters
mse_t_a = []
mse_v_a = []
best_mse_train = 1.14
best_mse_valid = 1.16

eta = [10e-05,10e-06,10e-07]
beta = [0.001,10e-04,10e-05,10e-06]
epsilon = [10e-05,10e-06,10e-07]
for n0 in eta:
    for b in beta:
        for e in epsilon:
            print(n0,b,e)
            w_g = w_gradient(x_train,y_train,n0,b,e)
            mse_train = MSE(x_train, w_g, y_train)
            mse_valid = MSE(x_valid, w_g, y_valid)
            if (mse_train < best_mse_train):
                best_mse_train = mse_train
                mse_t_a.append((n0,b,e,mse_train))
            if (mse_valid < best_mse_valid):
                best_mse_valid = mse_valid
                mse_v_a.append((n0,b,e,mse_valid))

print("best train mse", mse_t_a)
print("best valid mse", mse_v_a)

for (n0,b,e,mse) in mse_t_a:
    for (v_n0,v_b,v_e,v_mse) in mse_v_a:
        if n0==v_n0 and b==v_b and e==v_e:
            print(n0,b,e,mse)
            print(v_n0,v_b,v_e,v_mse)
'''
#end
