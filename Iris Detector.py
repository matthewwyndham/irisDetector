# Created by Matt Wyndham
from random import shuffle
import iris_loader
import HardCoded
import math


def learn(num_neighbors=1, should_i_print=True):
    class dataSet:
        data = []
        target = []

    iris = dataSet()

    iris.data, iris.target = iris_loader.load()

    # randomize order
    shuffled_data = []
    shuffled_target = []
    index_shuffler = [n for n in range(len(iris.data))]
    shuffle(index_shuffler)

    for index in index_shuffler:
        shuffled_data.append(iris.data[index])
        shuffled_target.append(iris.target[index])

    # split the data
    training_data = []
    training_target = []

    test_data = []
    test_target = []

    for splitter in range(len(iris.data)):
        if splitter < (len(iris.data) * .7):
            training_data.append(shuffled_data[splitter])
            training_target.append(shuffled_target[splitter])

        if splitter >= (len(iris.data) * .7):
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

    class ID3Tree:
        class Node(object):
            def __init__(self, data):
                self.data = data
                self.children = []

            def add_child(self, obj):
                self.children.append(obj)

        def fit(self, t_data, t_target):
            pass

        def predict(self, targets):
            predictions = []
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
        print(accuracy, "% accuracy")

    return accuracy

if __name__ == '__main__':

    # try different numbers of neighbors
    for i in range(1,99):
        print("Neighbors:", i)
        average = 0
        # take the average of 20 tries
        for _ in range(20):
            average += learn()
        average /= 20
        print("average:", average, "%")