# Created by Matt Wyndham
# Loads the iris data from a text file


def load():
    file = open('iris/iris.data')
    data_target_text = file.readlines()
    loader_data = []
    loader_target = []
    for i in range(len(data_target_text)):
        if data_target_text[i] != '\n':  # gotta avoid that final blank line for some reason
            values = data_target_text[i].split(',')
            numbers = []
            for m in range(len(values) - 1):
                numbers.append(float(values[m]))
            loader_data.append(numbers)
            if values[(len(values) - 1)] == "Iris-setosa\n":
                loader_target.append(0)
            if values[(len(values) - 1)] == 'Iris-versicolor\n':
                loader_target.append(1)
            if values[(len(values) - 1)] == 'Iris-virginica\n':
                loader_target.append(2)

    return loader_data, loader_target

if __name__ == '__main__':
    data, target = load()
    print(data)
    print(target)
    print("Data size:", len(data))
    print("Target size:", len(target))
