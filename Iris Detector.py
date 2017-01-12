#import the iris data
from sklearn import datasets
iris = datasets.load_iris()

print(iris.data)
print(iris.target)
print(iris.target_names)

for i in iris.data:
    print(i)

# randomize the data
from random import shuffle
iris_datum = iris.data
shuffle(iris_datum)

print(iris_datum)

# print("------------------")
#
## test shuffle of list of lists
# test_list = [[1, 2], [3, 4], [5, 6], [7,8]]
#
# print(test_list)
# shuffle(test_list)
# print(test_list)