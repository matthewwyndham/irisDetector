import math

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