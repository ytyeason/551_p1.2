#             IMPORT            #
import matplotlib.pyplot as plt
import numpy as np
import json # we need to use the JSON package to load the data, since the data is stored in JSON format
from datetime import datetime

# task 1
with open("proj1_data.json") as fp:
    data = json.load(fp)
fp.close()

with open("proj1_data.json") as fp:
    data2 = json.load(fp)
fp.close()

frequency_map = {}

def build_freq_map(data_set):
    for item in data_set:
        for word in item['text']:
                if word in frequency_map:
                    frequency_map[word] = frequency_map[word] + 1
                else:
                    frequency_map[word] = 1

def pre_process(data_set, nb_of_top_words=160, build=True):
    #preprocess is_root and text
    for item in data_set:
        if item['is_root'] == True:
            item['is_root'] = 1
        else:
            item['is_root'] = 0
        item['text'] = item['text'].lower().split()

    if build:
        build_freq_map(data_set)

    #sort the map by key in dscending order
    frequency_map_sorted = sorted(frequency_map.items(), key=lambda kv: kv[1], reverse = True)

    #first get the top 160 item of the map, then get the keys into a list
    most_frequent_word = [i[0] for i in frequency_map_sorted[0:nb_of_top_words]]


    for item in data_set:
        x_counts = [0.0]*nb_of_top_words
        for word in item['text']:
            if word in most_frequent_word:
                index = most_frequent_word.index(word)
                x_counts[index] = x_counts[index] + 1.0
        item['w_counts'] = x_counts

    return data_set

def getXandY(data, with_text_feature=1, with_extra_features=0):
    x_set = []
    y_set = []
    for item in data:
        if (with_text_feature == 1 and with_extra_features == 1):
            x_others = [1, np.log(item['children'] + 1),item['children'], np.power(item['children'],2),item['controversiality'], item['is_root']] # 2 other features + bias
            x_set.append(x_others + item['w_counts'])  #add each data set
        elif (with_text_feature == 1 and with_extra_features == 0): # only text feature
            x_others = [1,item['children'], item['controversiality'], item['is_root']] # bias
            x_set.append(x_others + item['w_counts'])  #add each data set
        elif (with_text_feature == 0 and with_extra_features == 0): # only 3 simple features
            x_set.append([1, item['children'], item['controversiality'], item['is_root']])

        y_set.append([item['popularity_score']])

    return np.array(x_set), np.array(y_set)

# task 2
def w_closed(X,Y):
    temp1 = np.dot(X.T,X)
    temp2 = np.dot(X.T,Y)

    w = np.dot(np.linalg.inv(temp1),temp2)
    return w

err_decay = []

# epsilon should be <= 10^-6, eta < 10^-5, beta < 0.001
def w_gradient(X,Y,eta=1e-08,beta=1e-06,e=1e-06):
    dim = np.array(X).shape[1]
    weight = np.random.random((dim,1))
    diff = np.ones((dim,1))

    # for faster computation
    xtx = np.dot(X.T, X)
    xty = np.dot(X.T, Y)

    # test convergence
    past_diff = 10

    i = 1
    while np.linalg.norm(diff,2) > e:
        alpha = eta / (1 + beta * i)

        diff = 2*alpha*(np.dot(xtx, weight)-xty)
        weight = weight - diff

        # test convergence
        if (i==1):
            past_diff = np.linalg.norm(diff,2)
        if (i == 2 and past_diff <= np.linalg.norm(diff,2)):
            print("   DO NOT CONVERGE")
            return np.ones((dim,1))
        err_decay.append(np.linalg.norm(diff,2))
        i = i + 1
    print("   ", i," iterations")
    return weight

def MSE(X,w,y):
    xw = X @ w
    return ((y - xw).T @ (y-xw) / len(y))[0][0]

# task 3

training_set = data[0:10000]
validation_set = data[10000:11000]
testing_set = data[11000:12000]

training_set2 = data2[0:10000]
validation_set2 = data2[10000:11000]
testing_set2 = data2[11000:12000]

print ("TASK 3 -------------------------------------------------------------")
# get 60 words training set
training_set60 = pre_process(training_set,60)
validation_set60 = pre_process(validation_set,60,False)

frequency_map = {}

# get 160 words
training_set160 = pre_process(training_set2,160)
validation_set160 = pre_process(validation_set2,160,False)
testing_set160 = pre_process(testing_set2,160,False)

fin = open("top_160.txt",'w')
fin.write(str(sorted(frequency_map.items(), key=lambda kv: kv[1],reverse = True)[0:160]))
fin.close()

# get x and y with no text features
x_train, y_train = getXandY(training_set60,0,0)
x_valid, y_valid = getXandY(validation_set60,0,0)


# get x and y with 60 text features only
x_train60, y_train60 = getXandY(training_set60,1,0)
x_valid60, y_valid60 = getXandY(validation_set60,1,0)

# get x and y with 60 text features only + 2 extra features
x_train62, y_train62 = getXandY(training_set60,1,1)
x_valid62, y_valid62 = getXandY(validation_set60,1,1)

# get x and y with 160 text features
x_train160, y_train160 = getXandY(training_set160,1,0)
x_valid160, y_valid160 = getXandY(validation_set160,1,0)

# get x and y with 160 text features + 2 extra features
x_train162, y_train162 = getXandY(training_set160,1,1)
x_valid162, y_valid162 = getXandY(validation_set160,1,1)


print('********** visualize data ***********')
child2 = [np.power(lst[1],2) for lst in x_train]
pop = []
for lst in y_train:
    pop.append(lst)

plt.title("Number of Children Squared vs. Popularity")
plt.plot(child2,pop,'g^')
plt.xlabel('Number of Children Squared')
plt.ylabel('Popularity')
plt.show()

print("********** decay of error ***********")
w_gd_162 = w_gradient(x_train162,y_train162)
plt.title('Decay of Error in Gradient Descent per Interation')
plt.plot(err_decay[0:1000], 'g^')
plt.show()

print("********** Testing with 3 simple features **********")
startTime = datetime.now()
w_closed_3Simple = w_closed(x_train,y_train)
endTime = datetime.now()
print("closed form: ")
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train, w_closed_3Simple, y_train))
print("   MSE valid: ", MSE(x_valid, w_closed_3Simple, y_valid))

print()

# gradient using different learning rates
eta = [10e-05,10e-06,10e-07]
beta = [0.001,1e-04,1e-05,1e-06]

for n0 in eta:
    for b in beta:
        startTime = datetime.now()
        w_gd_3Simple = w_gradient(x_train,y_train,n0,b)
        endTime = datetime.now()
        print('gradient with eta = ',n0, ' beta = ', b, ': ')
        print("   runtime: ", endTime-startTime, "minutes")
        print("   MSE train: ", MSE(x_train, w_gd_3Simple, y_train))
        print("   MSE valid: ", MSE(x_valid, w_gd_3Simple, y_valid))

print()

print('********** test which values to give to hyperparameters with 162 features **********')
mse_t_a = []
mse_v_a = []
best_mse_train = 1.14
best_mse_valid = 1.16

# epsilon should be <= 10^-6, eta < 10^-5, beta < 0.001
'''
eta = [1e-05,1e-06,1e-07,1e-08]
beta = [0.001,1e-04,1e-05,1e-06,1e-07]
epsilon = [1e-05,1e-06]
for n0 in eta:
    for b in beta:
        for e in epsilon:
            print(n0,b,e)
            w_g = w_gradient(x_train162,y_train162,n0,b,e)
            mse_train = MSE(x_train162, w_g, y_train162)
            mse_valid = MSE(x_valid162, w_g, y_valid162)
            if (eta == 1e-08):
                print(mse_train)
                print(mse_valid)
            if (mse_train < best_mse_train):
                best_mse_train = mse_train
                mse_t_a.append((n0,b,e,mse_train))
            if (mse_valid < best_mse_valid):
                best_mse_valid = mse_valid
                mse_v_a.append((n0,b,e,mse_valid))

print(mse_t_a)
print(mse_v_a)
'''
print()

print('********** Compare 3 models using closed-form **********')
print('model with only 3 simple features and no text features: ')
startTime = datetime.now()
w_closed_3Simple = w_closed(x_train,y_train)
endTime = datetime.now()
print("3 simple features closed form: ")
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train, w_closed_3Simple, y_train))
print("   MSE valid: ", MSE(x_valid, w_closed_3Simple, y_valid))

print()

print('model with only the top-60 words: ')
startTime = datetime.now()
w_closed_top60 = w_closed(x_train60,y_train60)
endTime = datetime.now()
print("top-60 words closed form: ")
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train60, w_closed_top60, y_train60))
print("   MSE valid: ", MSE(x_valid60, w_closed_top60, y_valid60))

print()

print('model with the full 160 word occurence: ')
startTime = datetime.now()
w_closed_top160 = w_closed(x_train160,y_train160)
endTime = datetime.now()
print("top-160 words closed form: ")
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train160, w_closed_top160, y_train160))
print("   MSE valid: ", MSE(x_valid160, w_closed_top160, y_valid160))

print()

print('********** adding two new features **********')

print("top-160 words + 2 features closed form: ")
startTime = datetime.now()
w_closed_162 = w_closed(x_train162,y_train162)
endTime = datetime.now()
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train162, w_closed_162, y_train162))
print("   MSE valid: ", MSE(x_valid162, w_closed_162, y_valid162))

print()

print('top-60 words + 2 features closed form: ')
startTime = datetime.now()
w_closed_top62 = w_closed(x_train62,y_train62)
endTime = datetime.now()
print("top-60 words closed form: ")
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train62, w_closed_top62, y_train62))
print("   MSE valid: ", MSE(x_valid62, w_closed_top62, y_valid62))

'''
print("top-160 words + 2 features gradient descent: ")
startTime = datetime.now()
w_gd_162 = w_gradient(x_train162,y_train162)
endTime = datetime.now()
print("   runtime: ", endTime-startTime, "minutes")
print("   MSE train:", MSE(x_train162, w_gd_162, y_train162))
print("   MSE valid: ", MSE(x_valid162, w_gd_162, y_valid162))
'''
print()

print("********** run 162 features' model on test set **********")
x_test162, y_test162 = getXandY(testing_set160,1,1)

print("the weight is: \n", w_closed_162)
print("MSE test: ", MSE(x_test162, w_closed_162, y_test162))
