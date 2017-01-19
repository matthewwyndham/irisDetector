# Created by Matt Wyndham
from random import shuffle
import dataLoader
import math


def learn(num_neighbors=1, should_i_print=True):
    class IrisHolder:
        data = []
        target = []

    iris = IrisHolder()

    iris.data, iris.target = dataLoader.load()

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

    # the black box
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

        def getDist(self, data1, data2):
            sum_of_squares = 0
            for i in range(len(data1)):

                sum_of_squares += pow(data1[i] - data2[i], 2)
            distance = math.sqrt(sum_of_squares)
            return distance

        def fit(self, t_data, t_targets):
            self.target = t_targets
            self.data = t_data

        def predict(self, test_data, k = 1):
            predictions = []

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


    # now get intelligent
    #GLADos = HardCoded()
    GLADos = WyndhammerKNN()

    GLADos.fit(training_data, training_target)
    predicted_targets = GLADos.predict(test_data, num_neighbors)

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
    # import the iris data

    #from sklearn import datasets
    #iris = datasets.load_iris()

    # try different numbers of neighbors
    for i in range(1,7):
        print("Neighbors:", i)
        average = 0
        # take the average of 20 tries
        for _ in range(20):
            average += learn(i, False)
        average /= 20
        print("average:", average, "%")