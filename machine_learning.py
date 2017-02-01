# Created by Matt Wyndham
from random import shuffle
import iris_loader
import lenses_loader
import vote_loader
import math

# only use for numeric data
def nominalize(data):
    # this will turn numeric data into nominal data
    # without reordering the data.
    for attribute in range(len(data[0])):
        # these should change to the actual max/min
        minimum_value = data[0][attribute]
        maximum_value = data[0][attribute]
        for row in data:
            if row[attribute] < minimum_value:
                minimum_value = row[attribute]
            if row[attribute] > maximum_value:
                maximum_value = row[attribute]
        # this only splits the data into three different groups
        # adding more groups should be easy
        split_value = (maximum_value + minimum_value) / 3
        for row in data:
            if row[attribute] < split_value:
                row[attribute] = 1
            elif row[attribute] < (split_value * 2):
                row[attribute] = 2
            else:
                row[attribute] = 3
    return data

# this is where the actual machine learning algorithms are stored
def learn(should_i_print=True):
    class dataSet:
        data = []
        target = []

    dataSet1 = dataSet()

    #########################################################
    ###   This is where you pick the data you will load   ###
    #########################################################

    ### Nominal Datasets ###


    ## Votes ##
    dataSet1.data, dataSet1.target = vote_loader.load() # nominal

    ## Lenses ##
    #dataSet1.data, dataSet1.target = lenses_loader.load() # nominal


    ### Numeric Datasets ###


    ## Iris ##
    #dataSet1.data, dataSet1.target = iris_loader.load() # numeric


    ### Split numeric datasets into nominal datasets ###
    ### Useful for the ID3 algorithm ###
    #dataSet1.data = nominalize(dataSet1.data)

    ###########################################
    ###   End the data picking section   ###
    ###########################################

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

    ######################################
    # This is where the ID3 stuff begins #
    ######################################

    # determines the entropy of one item. Use this multiple times per grouping
    def calc_entropy(p):
        if p != 0:
            return -p * math.log(p, 2)
        else:
            return 0

    # the class that can call itself
    class DecisionTree:
        def __init__(self):
            self.target = None # used by the prediction algorithm
            self.children = [] # predict
            self.attribute = None # predict
            self.isLeaf = None # predict

            self.data = []  # used by the fit algorithm
            self.numberOfTargets = None  # fit
            self.entropy = None # fit
            self.descendantNumber = 0 # fit

        def display(self):
            print('Attribute:', self.attribute, 'Target:', self.target)
            for x in self.children:
                for m in range(self.descendantNumber):
                    print('| ', end='')
                x.display()

        def find(self, target):
            if self.isLeaf == True:
                return self.target
            else:
                for child in self.children:
                    if target[self.attribute] == child.data[0][self.attribute]:
                        return child.find(target)


        def fit(self, num_values):
            # in case of error
            if len(self.data) == 0:
                self.target = 1
                self.isLeaf = True
                return

            sameTarget = True
            for row in range(len(self.data) - 1):
                if self.data[row][num_values] != self.data[row + 1][num_values]:
                    sameTarget = False
            if sameTarget == True:
                self.isLeaf = True
                self.target = self.data[0][num_values]
                return
            elif self.descendantNumber > 10: # length cap
                self.isLeaf = True
                self.target = self.data[0][num_values]
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
                    numTargetsPerChild = []
                    entropyPerChild = []
                    for c in FakeChildren:
                        targets = [0 for x in range(self.numberOfTargets)]
                        for x in range(len(c.data)):
                            targets[c.data[x][len(c.data[x]) - 1] - 1] += 1
                        total = 0
                        for i in targets:
                            total += i
                        numTargetsPerChild.append(total)
                        theEntropy = 0
                        for everyN in targets:
                            theEntropy += calc_entropy(everyN/total)
                        entropyPerChild.append(theEntropy)
                    weightedEntropy = 0
                    totalTargets = 0
                    for target in numTargetsPerChild:
                        totalTargets += target
                    for item in range(len(entropyPerChild)):
                        weightedEntropy += (entropyPerChild[item]*(numTargetsPerChild[item]/totalTargets))
                    entropyValues.append(weightedEntropy)
                # pick best option
                bestAttribute = 0
                for i in range(len(entropyValues)):
                    if entropyValues[i] < entropyValues[bestAttribute]:
                        bestAttribute = i

                # set that to this node's attribute
                self.attribute = bestAttribute

                # split data by that attribute
                possibleValues = []
                for i in range(len(self.data)):
                    if len(possibleValues) == 0:
                        possibleValues.append(self.data[i][self.attribute])
                    else:
                        alreadyThere = False
                        for j in range(len(possibleValues)):
                            if self.data[i][self.attribute] == possibleValues[j]:
                                alreadyThere = True
                        if alreadyThere != True:
                            possibleValues.append(self.data[i][self.attribute])

                # pass portion of data on to each child
                for v in possibleValues:
                    currentChild = DecisionTree()
                    for i in range(len(self.data)):
                        if self.data[i][self.attribute] == v:
                            currentChild.data.append(self.data[i])
                        currentChild.descendantNumber = self.descendantNumber + 1
                        currentChild.numberOfTargets = self.numberOfTargets
                    self.children.append(currentChild)

                # make each child fit themselves
                for c in self.children:
                    c.fit(num_values)
                return

    # initializes and wraps the actual nodes
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
                # follow the tree to a leaf node recursively
                predictions.append(self.root.find(targets[item]))

            self.root.display()

            return predictions

    ####################
    # End of ID3 Stuff #
    ####################

    # Naive Bayes
    class NaiveBayes:
        def fit(self, train_data, train_targets):
            # sum up everything
            # divide by the totals
            pass
        def predict(self, test_data):
            predictions = []
            # find the probability of each attribute
            # add it all up
            # find the highest of the p(attributes|each target value)
            # return that target value
            return predictions

    ##############################
    # choose your algorithm here #
    ##############################
    #GLADos = HardCoded()
    #GLADos = WyndhammerKNN()
    GLADos = ID3Tree()
    #GLADos = NaiveBayes() # DOES NOTHING #

    ## now the program checks things for you ##
    GLADos.fit(training_data, training_target)
    predicted_targets = GLADos.predict(test_data)

    # tell me your predictions!
    correct_predictions = 0
    for i in range(len(predicted_targets)):
        if predicted_targets[i] == test_target[i]:
            correct_predictions += 1

    # I check your accuracy
    accuracy = (correct_predictions / len(test_target)) * 100
    if should_i_print:
        print('Guess:', predicted_targets)
        print('Actual:', test_target)
        print(accuracy, "% accuracy")

    return accuracy

if __name__ == '__main__':
    average_accuracy = 0
    attempts = 0
    for i in range(15):
        print("Attempt: ", i,)
        average_accuracy += learn()
        attempts += 1
        print()
    print()
    print(average_accuracy / attempts)