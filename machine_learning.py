# Created by Matt Wyndham
from random import shuffle
import iris_loader
import lenses_loader
import math


def learn(should_i_print=True):
    class dataSet:
        data = []
        target = []

    dataSet1 = dataSet()
#    dataSet1.data, dataSet1.target = iris_loader.load()
    dataSet1.data, dataSet1.target = lenses_loader.load()

    # randomize order
    shuffled_data = []
    shuffled_target = []
    index_shuffler = [n for n in range(len(dataSet1.data))]
    shuffle(index_shuffler)

    for index in index_shuffler:
        shuffled_data.append(dataSet1.data[index])
        shuffled_target.append(dataSet1.target[index])

    # split the data
    training_data = []
    training_target = []

    test_data = []
    test_target = []

    for splitter in range(len(dataSet1.data)):
        if splitter < (len(dataSet1.data) * .7):
            training_data.append(shuffled_data[splitter])
            training_target.append(shuffled_target[splitter])

        if splitter >= (len(dataSet1.data) * .7):
            test_data.append(shuffled_data[splitter])
            test_target.append(shuffled_target[splitter])

    class HardCoded:
        target = []
        data = []

        def fit(self, t_data, t_targets):
            self.target = t_targets
            self.data = t_data

        def predict(self, test_data):
            predictions = []
            for i in range(len(test_data)):
                predictions.append(0)
            return predictions

    class WyndhammerKNN:
        target = []
        data = []
        mins = []
        maxes = []

        def getDist(self, data1, data2):
            sum_of_squares = 0
            for i in range(len(data1)):
                sum_of_squares += pow(data1[i] - data2[i], 2)
            distance = math.sqrt(sum_of_squares)
            return distance

        def fit(self, t_data, t_targets):
            self.target = t_targets
            self.data = t_data
            # normalize
            for dataType in range(len(t_data[0])):
                self.mins.append(t_data[0][dataType])
                self.maxes.append(t_data[0][dataType])
                for p in range(len(t_data)):
                    if t_data[p][dataType] < self.mins[dataType]:
                        self.mins[dataType] = t_data[p][dataType]
                    if t_data[p][dataType] > self.maxes[dataType]:
                        self.maxes[dataType] = t_data[p][dataType]
            for p in range(len(t_data)):
                for set in range(len(t_data[p])):
                    old = t_data[p][set]
                    t_data[p][set] = (old - self.mins[set]) / (self.maxes[set] - self.mins[set])

        def predict(self, test_data, k=1):
            predictions = []

            # normalize test data (doesn't handle outliers in the test data very well though...)
            for p in range(len(test_data)):
                for set in range(len(test_data[p])):
                    old = test_data[p][set]
                    test_data[p][set] = (old - self.mins[set]) / (self.maxes[set] - self.mins[set])

            # where the magic happens
            for i in range(len(test_data)):
                distances = []
                for j in range(len(self.data)):
                    key = [self.getDist(self.data[j], test_data[i]), j]
                    distances.append(key)
                sorted_distances = sorted(distances, key=lambda x: x[0])

                # find K nearest neighbors and upvote the type
                #        0  1  2
                votes = [0, 0, 0]
                for num in range(k):
                    if self.target[sorted_distances[num][1]] == 0:
                        votes[0] += 1
                    elif self.target[sorted_distances[num][1]] == 1:
                        votes[1] += 1
                    else:
                        votes[2] += 1

                choice = 0
                if votes[0] < votes[1]:
                    if votes[1] < votes[2]:
                        choice = 2
                    else:
                        choice = 1
                elif votes[0] < votes[2]:
                    choice = 2

                predictions.append(choice)

            return predictions

    # This is where the ID3 stuff begins

    def calc_entropy(p):
        if p != 0:
            return -p * math.log(p, 2)
        else:
            return 0

    class DecisionTree:
        def __init__(self):
            self.data = []
            self.numberOfTargets = None
            self.children = []
            self.attribute = None
            self.isLeaf = None
            self.entropy = None

        def fit(self, num_values):
            sameTarget = True
            for row in range(len(self.data) - 1):
                if self.data[row][num_values] != self.data[row + 1][num_values]:
                    sameTarget = False
            if sameTarget == True:
                isLeaf = True
                return
            else:
                # make split children for every attribute
                entropyValues = []
                for current in range(len(self.data[0]) - 1):
                    # every possible value of the current attribute according to the training set
                    FakeChildren = []
                    possibleValues = []
                    for i in range(len(self.data)):
                        if len(possibleValues) == 0:
                            possibleValues.append(self.data[i][current])
                        else:
                            alreadyThere = False
                            for j in range(len(possibleValues)):
                                if self.data[i][current] == possibleValues[j]:
                                    alreadyThere = True
                            if alreadyThere != True:
                                possibleValues.append(self.data[i][current])
                    for v in possibleValues:
                        currentChild = DecisionTree()
                        for i in range(len(self.data)):
                            if self.data[i][current] == v:
                                currentChild.data.append(self.data[i])
                        FakeChildren.append(currentChild)
                    # find entropy of each
                    for c in FakeChildren:
                        targets = [0 for x in range(self.numberOfTargets)]
                        for x in range(len(c.data)):
                            targets[c.data[x][len(c.data[x]) - 1] - 1] += 1
                        total = 0
                        for i in targets:
                            total += i

                        # perhaps this has gotten too complicated...

                # pick best option
                # set that to this node's attribute
                # split data by that attribute
                # pass portion of data on to each child
                # make each child fit themselves
                return

    class ID3Tree:
        num_values = 0
        # contains all the test data
        treeMakingData = []
        root = DecisionTree()

        def fit(self, t_data, t_target):
            for value in t_data[0]:
                self.num_values += 1
            for row in range(len(t_data)):
                current = t_data[row]
                current.append(t_target[row])
                self.treeMakingData.append(current)
            self.root.data = self.treeMakingData
            differentTargets = []
            for row in range(len(t_target)):
                if len(differentTargets) == 0:
                    differentTargets.append(t_target[row])
                else:
                    found = False
                    for t in range(len(differentTargets)):
                        if t_target[row] == differentTargets[t]:
                            found = True
                    if found == False:
                        differentTargets.append(t_target[row])
            self.root.numberOfTargets = len(differentTargets)
            self.root.fit(self.num_values)

        def predict(self, targets):
            predictions = []
            for item in range(len(targets)):
                predictions.append(1)
            return predictions

    # choose your algorithm here
    GLADos = ID3Tree()

    GLADos.fit(training_data, training_target)
    predicted_targets = GLADos.predict(test_data)

    # tell me your ideas
    correct_predictions = 0
    for i in range(len(predicted_targets)):
        if predicted_targets[i] == test_target[i]:
            correct_predictions += 1

    # I check your accuracy
    accuracy = (correct_predictions / len(test_target)) * 100
    if should_i_print:
        print(predicted_targets)
        print(test_target)
        print(accuracy, "% accuracy")

    return accuracy

if __name__ == '__main__':
    for i in range(15):
        print("Attempt: ", i,)
        learn()
        print()