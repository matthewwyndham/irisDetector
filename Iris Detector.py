# Created by Matt Wyndham

# import the iris data
from sklearn import datasets
from random import shuffle

iris = datasets.load_iris()

# randomize order
shuffled_data = []
shuffled_target = []
index_shuffler = [i for i in range(len(iris.data))]
shuffle(index_shuffler)

for i in index_shuffler:
    shuffled_data.append(iris.data[i])
    shuffled_target.append(iris.target[i])

# split the data
training_data = []
training_target = []

test_data = []
test_target = []

for i in range(len(iris.data)):
    if i < (len(iris.data) * .7):
        training_data.append(shuffled_data[i])
        training_target.append(shuffled_target[i])

    if i >= (len(iris.data) * .7):
        test_data.append(shuffled_data[i])
        test_target.append(shuffled_target[i])

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

# now get intelligent
GLADos = HardCoded()

GLADos.fit(training_data, training_target)
predicted_targets = GLADos.predict(test_data)

# tell me your ideas
correct_predictions = 0
for i in range(len(predicted_targets)):
    if predicted_targets[i] == test_target[i]:
        correct_predictions += 1

# I check your accuracy
accuracy = (correct_predictions / len(test_target)) * 100

print(accuracy, "% accuracy")