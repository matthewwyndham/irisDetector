# Loads the iris data from a text file
def load():
    file = open('iris.data')
    data_target_text = file.readlines()
    #print(data_target_text)
    #print("initial list size:", len(data_target_text))
    data = []
    target = []
    for i in range(len(data_target_text)):
        if data_target_text[i] != '\n': # gotta avoid that final blank line for some reason
            values = data_target_text[i].split(',')
            numbers = []
            for i in range(len(values) - 1):
                numbers.append(float(values[i]))
            data.append(numbers)
            if values[(len(values) - 1)] == "Iris-setosa\n":
                target.append(0)
            if values[(len(values) - 1)] == 'Iris-versicolor\n':
                target.append(1)
            if values[(len(values) - 1)] == 'Iris-virginica\n':
                target.append(2)

    return data, target

if __name__ == '__main__':
    data, target = load()
    print(data)
    print(target)
    print("Data size:", len(data))
    print("Target size:", len(target))