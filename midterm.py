# %%
import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model
import random
import statistics

# %%
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N

# %%
answers = {}

# %%
z = gzip.open("/Users/tiyang/Downloads/MGTA461_Midterm2023/train.json.gz")

# %%
dataset = []
for l in z:
    d = eval(l)
    dataset.append(d)

# %%
z.close()

# %%
### Question 1

# %%
def MSE(y, ypred):
    return numpy.mean((y - ypred)**2)

# %%
def MAE(y, ypred):
    return numpy.mean(numpy.abs(y - ypred))

# %%
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['userID'],d['gameID']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)
    
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['date'])
    
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['date'])

# %%
def feat1(d):
    return [d['hours']]

# %%
X = numpy.array([feat1(d) for d in dataset])
y = numpy.array([len(d['text']) for d in dataset])

# %%
mod = linear_model.LinearRegression()
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q1 = MSE(y, predictions)
theta_1 = mod.coef_[0]

# %%
answers['Q1'] = [theta_1, mse_q1]

# %%
assertFloatList(answers['Q1'], 2)

print(answers['Q1'])

# %%
### Question 2

# %%
hours_list = [d['hours'] for d in dataset]
median_hours = statistics.median(hours_list)

# %%
def feat2(d):
    hours = d['hours']
    return [
        1,  # for the intercept term, θ0
        hours,  # θ1 * (hours)
        math.log2(hours + 1),  # θ2 * log2(hours + 1)
        math.sqrt(hours),  # θ3 * sqrt(hours)
        1 if hours > median_hours else 0  # θ4 * δ(hours > median)
    ]

# %%
X = [feat2(d) for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q2 = MSE(y, predictions)

# %%
answers['Q2'] = mse_q2

# %%
assertFloat(answers['Q2'])

print(answers['Q2'])

# %%
### Question 3

# %%
def feat3(d):
    hours = d['hours']
    return [
        1,  # for the intercept term, θ0
        1 if hours > 1 else 0,    # θ1 * δ(h > 1)
        1 if hours > 5 else 0,    # θ2 * δ(h > 5)
        1 if hours > 10 else 0,   # θ3 * δ(h > 10)
        1 if hours > 100 else 0,  # θ4 * δ(h > 100)
        1 if hours > 1000 else 0  # θ5 * δ(h > 1000)
    ]

# %%
X = [feat3(d) for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse_q3 = MSE(y, predictions)

# %%
answers['Q3'] = mse_q3

# %%
assertFloat(answers['Q3'])

print(answers['Q3'])

# %%
### Question 4

# %%
def feat4(d):
    return [len(d['text'])]

# %%
X = [feat4(d) for d in dataset]
y = [d['hours'] for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=True)
mod.fit(X,y)
predictions = mod.predict(X)

# %%
mse = MSE(y, predictions)
mae = MAE(y, predictions)

# %%
explanation = "MAE may be more suitable when dealing with datasets with potential outliers, as it is less sensitive to extreme values than MSE."

# %%
answers['Q4'] = [mse, mae, explanation]

# %%
assertFloatList(answers['Q4'][:2], 2)

print(answers['Q4'])

# %%
### Question 5

# %%


# %%
y_trans = [math.log2(d['hours'] + 1) for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=True)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
mse_trans = MSE(y_trans, predictions_trans) # MSE using the transformed variable

# %%
y_array = numpy.array(y)

# %%
predictions_untrans = numpy.array([2**p - 1 for p in predictions_trans]) # Undoing the transformation

# %%
mse_untrans = MSE(y_array, predictions_untrans)

# %%
answers['Q5'] = [mse_trans, mse_untrans]

# %%
assertFloatList(answers['Q5'], 2)

print(answers['Q5'])

# %%
### Question 6

# %%
def feat6(d):
    hours = int(d['hours'])  # Get the integer part of the hours
    one_hot = [0]*100
    one_hot[min(hours, 99)] = 1  # Cap the feature at 99 hours
    return one_hot

# %%
X = [feat6(d) for d in dataset]
y = [len(d['text']) for d in dataset]

# %%
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]

# %%
from sklearn.linear_model import Ridge

# %%
models = {}
mses = {}
bestC = None
bestMSE = float('inf')

for c in [1, 10, 100, 1000, 10000]:
    models[c] = Ridge(alpha=c)
    models[c].fit(Xtrain, ytrain)

    predictions_valid = models[c].predict(Xvalid)

    mses[c] = MSE(yvalid, predictions_valid)

    if mses[c] < bestMSE:
        bestC = c
        bestMSE = mses[c]

# %%
best_model = models[bestC]

# %%
predictions_test = best_model.predict(Xtest)

# %%
mse_valid = bestMSE

# %%
mse_test = MSE(ytest, predictions_test)

# %%
answers['Q6'] = [bestC, mse_valid, mse_test]

# %%
assertFloatList(answers['Q6'], 3)

print(answers['Q6'])

# %%
### Question 7

# %%
times = [d['hours_transformed'] for d in dataset]
median = statistics.median(times)

# %%


# %%
notPlayed = [time for time in times if time < 1]
nNotPlayed = len(notPlayed)

# %%
answers['Q7'] = [median, nNotPlayed]

# %%
assertFloatList(answers['Q7'], 2)

print(answers['Q7'])

# %%
### Question 8

# %%
def feat8(d):
    return [len(d['text'])]

# %%
X = [feat8(d) for d in dataset]
y = [d['hours_transformed'] > median for d in dataset]

# %%
mod = linear_model.LogisticRegression(class_weight='balanced')
mod.fit(X,y)
predictions = mod.predict(X) # Binary vector of predictions

# %%
def rates(predictions, y):
    TP = sum(p and l for p, l in zip(predictions, y))
    TN = sum(not p and not l for p, l in zip(predictions, y))
    FP = sum(p and not l for p, l in zip(predictions, y))
    FN = sum(not p and l for p, l in zip(predictions, y))
    return TP, TN, FP, FN

# %%
TP, TN, FP, FN = rates(predictions, y)

# %%
P = sum(y) 
N = len(y) - P
BER = 0.5 * ((FP / N) + (FN / P))

# %%
answers['Q8'] = [TP, TN, FP, FN, BER]

# %%
assertFloatList(answers['Q8'], 5)

print(answers['Q8'])

# %%
### Question 9

# %%
probabilities = mod.predict_proba(X)[:,1]
sorted_by_prob = sorted(zip(probabilities, y), key=lambda x: x[0], reverse=True)

# %%
precs = []
recs = []

for i in [5, 10, 100, 1000]:
    sorted_data = sorted(zip(dataset, probabilities), key=lambda x: x[1], reverse=True)

    threshold = sorted_data[i-1][1] if i <= len(sorted_data) else sorted_data[-1][1]
    
    threshold_data = [(d, prob) for d, prob in sorted_data if prob >= threshold]
    
    TP = sum(1 for d, prob in threshold_data if d['hours_transformed'] > median)
    precision = TP / len(threshold_data)
    
    recall = TP / sum(1 for d in dataset if d['hours_transformed'] > median)
    
    precs.append(precision)
    recs.append(recall)

# %%
answers['Q9'] = precs

# %%
assertFloatList(answers['Q9'], 4)

print(answers['Q9'])

# %%
### Question 10

# %%
y_trans = [d['hours_transformed'] for d in dataset]

# %%
mod = linear_model.LinearRegression(fit_intercept=False)
mod.fit(X,y_trans)
predictions_trans = mod.predict(X)

# %%
ber_to_beat = answers['Q8'][4]

def calculate_ber(predictions, actual):
    TP = sum((p and a) for p, a in zip(predictions, actual))
    TN = sum((not p and not a) for p, a in zip(predictions, actual))
    FP = sum((p and not a) for p, a in zip(predictions, actual))
    FN = sum((not p and a) for p, a in zip(predictions, actual))
    P = sum(actual)  
    N = len(actual) - P  
    return 0.5 * ((FP / N) + (FN / P))

best_threshold = None
best_ber = float('inf')
for threshold in numpy.linspace(min(predictions_trans), max(predictions_trans), 1000):
    predictions_thresh = predictions_trans > threshold
    current_ber = calculate_ber(predictions_thresh, y)
    if current_ber < best_ber:
        best_ber = current_ber
        best_threshold = threshold

# %%
predictions_thresh = predictions_trans > best_threshold # Using a fixed threshold to make predictions

# %%
TP, TN, FP, FN = rates(predictions_thresh, y)

# %%
BER = 0.5 * ((FP / (FP + TN)) + (FN / (TP + FN)))

# %%
answers['Q10'] = [best_threshold, BER]

# %%
assertFloatList(answers['Q10'], 2)

print(answers['Q10'])

# %%
### Question 11

# %%
dataTrain = dataset[:int(len(dataset)*0.9)]
dataTest = dataset[int(len(dataset)*0.9):]

# %%
userMedian = defaultdict(list)
itemMedian = defaultdict(list)

# Compute medians on training data

# %%
for d in dataTrain:
    user, item, playtime = d['userID'], d['gameID'], d['hours']
    userMedian[user].append(playtime)
    itemMedian[item].append(playtime)

for user in userMedian:
    userMedian[user] = statistics.median(userMedian[user])

for item in itemMedian:
    itemMedian[item] = statistics.median(itemMedian[item])

first_item = dataTrain[0]['gameID']
first_user = dataTrain[0]['userID']

# %%
answers['Q11'] = [itemMedian['g35322304'], userMedian['u55351001']]

# %%
assertFloatList(answers['Q11'], 2)

print(answers['Q11'])

# %%
### Question 12

# %%
global_median = statistics.median([d['hours'] for d in dataTrain])

# %%
def f12(u,i):
    # Function returns a single value (0 or 1)
    if i in itemMedian:
        if itemMedian[i] > global_median:
            return 1
        else:
            return 0
    else:
        if u in userMedian and userMedian[u] > global_median:
            return 1
        else:
            return 0

# %%
preds = [f12(d['userID'], d['gameID']) for d in dataTest]

# %%
y = [1 if d['hours'] > global_median else 0 for d in dataTest]

# %%
accuracy = sum(1 for (pred, actual) in zip(preds, y) if pred == actual) / len(preds)

# %%
answers['Q12'] = accuracy

# %%
assertFloat(answers['Q12'])

print(answers['Q12'])

# %%
### Question 13

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
itemNames = {}

for d in dataset:
    user,item = d['userID'], d['gameID']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

# %%
def mostSimilar(i, func, N):
    similarities = []
    users = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue
        sim = func(users, usersPerItem[i2])
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:N]

# %%
ms = mostSimilar(dataset[0]['gameID'], Jaccard, 10)

# %%
answers['Q13'] = [ms[0][0], ms[-1][0]]

# %%
assertFloatList(answers['Q13'], 2)

print(answers['Q13'])

# %%
### Question 14

# %%
import math

# %%
def mostSimilar14(i, func, N):
    similarities = []
    users_i = usersPerItem[i]
    for i2 in usersPerItem:
        if i2 == i: continue  
        users_i2 = usersPerItem[i2]
        if not users_i or not users_i2:
            continue
        sim = func(i, i2)
        similarities.append((sim, i2))
    similarities.sort(reverse=True)
    return similarities[:N]

# %%
median_playtime = statistics.median([d['hours'] for d in dataset])

# %%
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = 1 if d['hours'] > median_playtime else -1# Set the label based on a rule
    ratingDict[(u,i)] = lab

# %%
def Cosine(i1, i2):
    # Between two items
    users_i1 = usersPerItem[i1]
    users_i2 = usersPerItem[i2]
    intersect = users_i1.intersection(users_i2)
    numerator = sum(ratingDict[(u, i1)] * ratingDict[(u, i2)] for u in intersect)
    denominator = math.sqrt(sum(ratingDict[(u, i1)]**2 for u in users_i1) * sum(ratingDict[(u, i2)]**2 for u in users_i2))
    return numerator / denominator if denominator != 0 else 0

# %%
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)

# %%
answers['Q14'] = [ms[0][0], ms[-1][0]]

# %%
assertFloatList(answers['Q14'], 2)

print(answers['Q14'])

# %%
### Question 15

# %%
ratingDict = {}

for d in dataset:
    u,i = d['userID'], d['gameID']
    lab = math.log(d['hours'] + 1)# Set the label based on a rule
    ratingDict[(u,i)] = lab

# %%
def CosineHours(i1, i2):
    users_i1 = usersPerItem[i1]
    users_i2 = usersPerItem[i2]
    inter_users = users_i1.intersection(users_i2)

    numerator = sum(ratingDict[(u, i1)] * ratingDict[(u, i2)] for u in inter_users)

    sum_sq_i1 = sum(ratingDict[(u, i1)]**2 for u in users_i1)
    sum_sq_i2 = sum(ratingDict[(u, i2)]**2 for u in users_i2)
    
    if sum_sq_i1 == 0 or sum_sq_i2 == 0:
        return 0
    
    return numerator / math.sqrt(sum_sq_i1 * sum_sq_i2)

# %%
ms = mostSimilar14(dataset[0]['gameID'], Cosine, 10)

# %%
answers['Q15'] = [ms[0][0], ms[-1][0]]

# %%
assertFloatList(answers['Q15'], 2)

print(answers['Q15'])

# %%
f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()

# %%



