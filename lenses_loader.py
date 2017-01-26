def load():
    file = open('lenses/lenses.data')
    data_target_text = file.readlines()
    loader_data = []
    loader_target = []
    for i in range(len(data_target_text)):
        if data_target_text[i] != '\n':  # gotta avoid that final blank line for some reason
            values = data_target_text[i].split()
            numbers = []
            for m in range(len(values) - 2):
                numbers.append(int(values[m + 1])) # this is to skip that first attribute that is just a counter of entries
            loader_data.append(numbers)
            loader_target.append(int(values[len(values) - 1]))

    return loader_data, loader_target

if __name__ == '__main__':
    data, target = load()
    print(data)
    print(target)
    print("Data size:", len(data))
    print("Target size:", len(target))