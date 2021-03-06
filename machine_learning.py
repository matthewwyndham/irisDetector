# Created by Matt Wyndham
from random import shuffle
from random import uniform
import iris_loader
import lenses_loader
import vote_loader
import diabetes_loader
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

# use to make numeric data range from 0 to 1 instead of it's current range
def normalize(data):
    mins = []
    maxes = []
    # find the mins and maxes
    for dataType in range(len(data[0])):
        mins.append(data[0][dataType])
        maxes.append(data[0][dataType])
        for p in range(len(data)):
            if data[p][dataType] < mins[dataType]:
                mins[dataType] = data[p][dataType]
            if data[p][dataType] > maxes[dataType]:
                maxes[dataType] = data[p][dataType]
    for p in range(len(data)):
        for set in range(len(data[p])):
            old = data[p][set]
            data[p][set] = (old - mins[set]) / (maxes[set] - mins[set])

    return data

# this is where the actual machine learning algorithms are stored
def learn(should_i_print=True):
    class dataSet:
        data = []
        target = []

    dataSet1 = dataSet()

    #########################################################
    ###   This is where you pick the data you will load   ###
    ###   TODO: Make the machine pick the data
    #########################################################

    ### Nominal Datasets ###


    ## Votes ##
    #dataSet1.data, dataSet1.target = vote_loader.load() # nominal

    ## Lenses ##
    #dataSet1.data, dataSet1.target = lenses_loader.load() # nominal


    ### Numeric Datasets ###


    ## Iris ##
    dataSet1.data, dataSet1.target = iris_loader.load() # numeric

    ## Diabetes ##
    #dataSet1.data, dataSet1.target = diabetes_loader.load()

    ### Split numeric datasets into nominal datasets ###
    ### Useful for the ID3 algorithm ###
    #dataSet1.data = nominalize(dataSet1.data)

    ### Normalize the data (i.e. make it range from 0 to 1) ###
    dataSet1.data = normalize(dataSet1.data)

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

    # Neural Network
    # Probably the most complicated one yet!
    # Only slightly more difficult than the ID3Tree though, and much easier to wrap my mind around #pun
    class Neuron:
        def __init__(self):
            self.weights = []
            self.threshold = 0
            self.error = None
            self.output = None
            self.number_of_inputs = 0
            self.bias_weight = -0.31415 # why not? -pi/10
            self.weights_are_set = False

            self.position_in_layer = 0

            self.learning_rate = 0.2

        # based on the input calculate the output
        def flowForward(self, iOutputs):
            if self.weights_are_set:
                # BIAS
                new_output = (-1 * self.bias_weight) # handle the bias of each node

                for node in range(self.number_of_inputs): # this needs to be set for each layer
                    new_output += iOutputs[node].output * self.weights[node]
                    # The SIGMOID function!
                new_output = 1 / (1 + math.e**(-new_output))
                self.output = new_output
                return
            else:
                self.setWeights(iOutputs)
                self.flowForward(iOutputs)

        # based on error of k, change weight between i and this (j)
        def backPropegate(self, kNodes):
            # do not worry about the bias node!
            weighted_errors = 0
            for node in range(len(kNodes)):
                weighted_errors += kNodes[node].weights[self.position_in_layer] * kNodes[node].error
            self.error = self.output * (1 - self.output) * (weighted_errors)

        def updateWeights(self, iNodes):
            # bias
            self.bias_weight -= self.learning_rate * self.error * -1
            # the rest
            for node in range(len(iNodes)):
                self.weights[node] -= self.learning_rate * self.error * iNodes[node].output

        # set up the weights at a small number for each input
        def setWeights(self, iOutputs):
            self.number_of_inputs = len(iOutputs)
            for i in range(self.number_of_inputs):
                self.weights.append(uniform(-0.5,0.5))
            self.weights_are_set = True

    class NeuralNet:
        def __init__(self):
            self.number_of_inputs = 0
            self.nodes = []
            self.different_targets = []

            # Multi-Layer Perceptron
            # keep track of all the nodes
            self.layers = []
            self.nodes_in_layer = []
            self.number_of_layers = 0

            # tells me when to stop training
            self.epoch_training_limit = 100
            self.epoch = 0

        def fit(self, training_data, training_target, npl = 4, hl = 2):
            # get validation data
            self.validation_data = []
            self.validation_targets = []
            # out of the training data
            self.data = []
            self.targets = []

            number_of_instances = len(training_target)
            for i in range(number_of_instances):
                if i < number_of_instances * .7:
                    self.data.append(training_data[i])
                    self.targets.append(training_target[i])
                else:
                    self.validation_data.append(training_data[i])
                    self.validation_targets.append(training_target[i])

            self.neurons_per_layer = npl
            self.hidden_layers = hl - 1 # sloppy variable: User wants 2 layers, 1 hidden, 1 output
            # we always make 1 output layer, so hidden layers minus 1.
            # hopefully they never put in 0 or less than zero...
            # oh and user is GLADos, so she'll have to be changed to actually pass in parameters.

            # find out the number of inputs
            for attribute in self.data[0]:
                self.number_of_inputs += 1

            # find out different targets
            # (this number will probably be smaller than the number of inputs)
            not_seen = True
            for instance in self.targets:
                # search through all the targets you've seen
                for target in self.different_targets:
                    if target == instance:
                        not_seen = False
                        break
                # if you still haven't seen the target, add it!
                if not_seen:
                    self.different_targets.append(instance)
                else: # set up for the next run
                    not_seen = True
            # sort the list for convenience
            self.different_targets.sort()

            # create some hidden layers & append to self.layers
            # input nodes
            inputLayer = []
            for i in range(len(self.data[0])):
                newInputNeuron = Neuron()
                newInputNeuron.output = self.data[0][i]
                inputLayer.append(newInputNeuron)

            # create the hidden layers
            firstLayer = True
            numNodes_thisLayer = 0
            for number in range(self.hidden_layers):
                if firstLayer:
                    firstLayer = False
                    nodelist = []
                    for node in range(self.neurons_per_layer):
                        nextNeuron = Neuron()
                        # set up the weights!
                        # and the output
                        nextNeuron.flowForward(inputLayer) # pass in the previous layer of nodes
                        nextNeuron.position_in_layer = numNodes_thisLayer
                        nodelist.append(nextNeuron)
                        numNodes_thisLayer += 1
                    self.layers.append(nodelist)
                    self.nodes_in_layer.append(numNodes_thisLayer)
                    numNodes_thisLayer = 0
                else:
                    nodelist = []
                    for node in range(self.neurons_per_layer):
                        nextNeuron = Neuron()
                        # set up the weights!
                        # and the output
                        nextNeuron.flowForward(self.layers[number - 1]) # pass in the previous layer of nodes
                        nextNeuron.position_in_layer = numNodes_thisLayer
                        nodelist.append(nextNeuron)
                        numNodes_thisLayer += 1
                    self.layers.append(nodelist)
                    self.nodes_in_layer.append(numNodes_thisLayer)
                    numNodes_thisLayer = 0

            self.number_of_layers = self.hidden_layers # probably unnecessary?

            # create the output layer
            nodelist = []
            numNodes_thisLayer = 0
            for value in range(len(self.different_targets)):
                nextNeuron = Neuron()
                # set up the weights!
                # and the output
                nextNeuron.flowForward(self.layers[len(self.layers) - 1])  # pass in the previous layer of nodes
                nextNeuron.position_in_layer = numNodes_thisLayer
                nodelist.append(nextNeuron)
                numNodes_thisLayer += 1
            self.layers.append(nodelist)
            self.nodes_in_layer.append(numNodes_thisLayer)
            self.number_of_layers += 1

            # Part III
            # accuracy over epochs
            accuracy_per_epoch = []
            # hard limit only train that many times
            for current_epoch in range(self.epoch_training_limit):
                # iterate through every data point
                for counter in range(len(self.data)):
                    # input nodes
                    inputLayer = []
                    for currentnode in range(len(self.data[counter])):
                        newInputNeuron = Neuron()
                        newInputNeuron.output = self.data[counter][currentnode]
                        inputLayer.append(newInputNeuron)

                    # run the input through the network
                    for layer in range(len(self.layers)):
                        if layer == 0:
                            for node in self.layers[layer]:
                                node.flowForward(inputLayer)
                        else:
                            for node in self.layers[layer]:
                                node.flowForward(self.layers[layer - 1])

                    # CALCULATE ERROR
                    # find target value this should be 1
                    target_node = 0
                    for target in range(len(self.different_targets)):
                        if self.targets[counter] == self.different_targets[target]:
                            target_node = target

                    # everything else is 0
                    for layer in range(len(self.layers) - 1, -1, -1):
                        if layer == len(self.layers) - 1:
                            # calculate the final layer error
                            for cnode in range(len(self.layers[layer])):
                                current_node = self.layers[layer][cnode]
                                actual_value = 0
                                if cnode == target_node:
                                    actual_value = 1
                                current_node.error = current_node.output * (1 - current_node.output) * (current_node.output - actual_value)
                                self.layers[layer][cnode].error = current_node.error # is this necessary?
                                                                        # I don't know but it makes me feel safe
                        else:
                            # calculate the hidden layers
                            for hnode in range(len(self.layers[layer])):
                                self.layers[layer][hnode].backPropegate(self.layers[layer + 1])

                    # UPDATE WEIGHTS
                    for layer in range(len(self.layers)):
                        if layer == 0:
                            for node in range(len(self.layers[layer])):
                                self.layers[layer][node].updateWeights(inputLayer)
                        else:
                            for node in range(len(self.layers[layer])):
                                self.layers[layer][node].updateWeights(self.layers[layer - 1])

                    # next instance in the training data

                # now test against the validation data
                predicted_targets = self.predict(self.validation_data)
                correct_predictions = 0
                for i in range(len(predicted_targets)):
                    if predicted_targets[i] == self.validation_targets[i]:
                        correct_predictions += 1
                accuracy = (correct_predictions / len(test_target)) * 100
                # add that to the list of accuracy
                accuracy_per_epoch.append(accuracy)

                # display a plot of the things

                # increment the epoch counter
            import matplotlib.pyplot as plt
            plt.plot(accuracy_per_epoch)
            plt.ylabel('Accuracy over time')
            plt.show()
            return

        def predict(self, test_data):
            predictions = []
            for instance in test_data:
                results = []

                # input nodes
                inputLayer = []
                for i in range(len(instance)):
                    newInputNeuron = Neuron()
                    newInputNeuron.output = instance[i]
                    inputLayer.append(newInputNeuron)

                # propagate the values
                first_layer = True
                for layer in range(len(self.layers)):
                    if first_layer: # first layer gets the inputs
                        first_layer = False
                        for node in range(len(self.layers[layer])):
                            self.layers[layer][node].flowForward(inputLayer)
                    elif layer is not len(self.layers) - 1: # next layers just reflow
                        for node in range(len(self.layers[layer])):
                            self.layers[layer][node].flowForward(self.layers[layer - 1])
                    else: # final layer should give us the outputs
                        for node in range(len(self.layers[layer])):
                            self.layers[layer][node].flowForward(self.layers[layer - 1])
                            results.append(self.layers[layer][node].output)

                # find the highest node in the values
                highest_value = 0 # the sigmoid function will always be greater than zero!
                value_location = -1
                for item in range(len(results)):
                    if results[item] > highest_value:
                        highest_value = results[item]
                        value_location = item
                predictions.append(self.different_targets[value_location]) # get the matching target


            # done and done!
            return predictions

    ##############################
    # choose your algorithm here #
    # TODO: Make the machine pick the best algorithm
    # TODO: Make the machine come up with new algorithms
    ##############################
    #GLADos = HardCoded()
    #GLADos = WyndhammerKNN()
    #GLADos = ID3Tree()
    #GLADos = NaiveBayes() # DOES NOTHING #
    GLADos = NeuralNet() # add extra parameters below to change the number of nodes.
    # for example: GLADos.fit(training_data, training_target, 8, 10) 8 nodes per layer, 10 layers

    ## now the program checks things for you ##
    GLADos.fit(training_data, training_target, 5, 3)
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
        print("Attempt: ", i + 1,)
        average_accuracy += learn()
        attempts += 1
        print()
    print("Total Average:")
    print(average_accuracy / attempts)