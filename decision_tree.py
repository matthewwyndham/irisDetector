class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []
        self.isLeaf = True

    def add_child(self, obj):
        if self.isLeaf == True:
            self.isLeaf = False
        self.children.append(obj)

    def isLeaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

class DecisionTree:
    root = Node()

    def fit(self, training_data, training_targets):
        pass

    def predict(self, test_data):
        pass