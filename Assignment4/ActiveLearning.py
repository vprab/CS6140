__author__ = 'Vishant'
from DecisionStumpAdaboost import *

#### LOADING DATA ####

# with open("spambase.data") as spambase:
#     spambase_data = [line.split(',') for line in spambase]
#
# spambase_data = np.array(spambase_data).astype(np.float)
# for i in range(np.shape(spambase_data)[0]):
#     if spambase_data[i][57] == 0:
#         spambase_data[i][57] = -1
#
# K = 10
# split_spambase = np.array([spambase_data[i::K] for i in range(K)])

#### ACTIVE LEARNING ####

not_learned = range(len(spambase_data))
five_percent = int(len(spambase_data) * 0.05)
two_percent = int(len(spambase_data) * 0.02)
fifty_percent = int(len(spambase_data) * 0.5)


def reservoir_sample(l, k):
    ans = l[0:k]

    for i in range(k, len(l)):
        j = random.choice(range(i+1))
        if j < k:
            ans[j] = l[i]

    return ans

training_set = reservoir_sample(not_learned, five_percent)
not_learned = [item for item in not_learned if item not in training_set]

count = 1
while len(training_set) < fifty_percent:
    train = np.array([row for ind, row in enumerate(spambase_data) if ind in training_set])
    not_learned_dict = {ind:float("inf") for ind in not_learned}

    train_stump = adaboost_decision_stump(train, 57, optimal_decision_stump, 100)

    for ind in not_learned:
        test_record_ind = spambase_data[ind][:57]
        not_learned_dict[ind] = apply_adaboost_decision_stump(test_record_ind, train_stump)

    not_learned_sorted = sorted(not_learned_dict, key=lambda ind: abs(not_learned_dict[ind]))

    print "Iteration %i - Training Error: %f" % (count, adaboost_decision_stump_error(spambase_data, 57, train_stump))

    count += 1

    training_set = training_set + not_learned_sorted[:two_percent]
    not_learned = [item for item in not_learned if item not in training_set]

