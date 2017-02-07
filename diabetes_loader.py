# Created by Matt Wyndham
# Loads the pima indian diabetes data from a text file


def load():
    file = open('pima_diabetes/pima-indians-diabetes.data')
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
            loader_target.append(int(values[(len(values) - 1)]))

    return loader_data, loader_target

if __name__ == '__main__':
    data, target = load()
    print(data)
    print("Data size:", len(data))
    print(target)
    print("Target size:", len(target))
