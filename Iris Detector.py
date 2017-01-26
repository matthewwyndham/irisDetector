# Created by Matt Wyndham
from random import shuffle
import iris_loader
import WyndhammerKNN
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

    # now get intelligent
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

    # try different numbers of neighbors
    for i in range(1,99):
        print("Neighbors:", i)
        average = 0
        # take the average of 20 tries
        for _ in range(20):
            average += learn(i, False)
        average /= 20
        print("average:", average, "%")