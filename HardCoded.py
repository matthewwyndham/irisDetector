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