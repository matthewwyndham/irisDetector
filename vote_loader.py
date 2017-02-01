# created by Matt Wyndham

def load():
    file = open('votes/house-votes-84.data')
    data_target_text = file.readlines()
    loader_data = []
    loader_target = []
    for i in range(len(data_target_text)):
        if data_target_text[i] != '\n':  # gotta avoid that final blank line for some reason
            values = data_target_text[i].split(',')
            vote_values = []
            for m in range(1, len(values) - 1):
                if values[m] == 'y':
                    vote_values.append(1)
                elif values[m] == 'n':
                    vote_values.append(2)
                else: # if there is a missing data point or anything weird it is value 3
                    vote_values.append(3)
            loader_data.append(vote_values)
            if values[0] == 'republican': # so a 1 means republican and a 2 means democrat
                loader_target.append(1)
            else:
                loader_target.append(2)

    return loader_data, loader_target

if __name__ == '__main__':
    data, target = load()
    print(data)
    print(target)
    print("Data size:", len(data))
    print("Target size:", len(target))
