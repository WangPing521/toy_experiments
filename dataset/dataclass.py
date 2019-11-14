from sklearn.utils import shuffle
from sklearn import datasets


class irisdata():

    def __init__(self, num1, num2, mode):
        super(irisdata, self).__init__()
        iris = datasets.load_iris()
        self.iris = iris
        self.lenth = len(iris.data)
        self.num1 = num1
        self.num2 = num2
        self.mode = mode

    def __getitem__(self, item):

        data, targets = shuffle(self.iris.data, self.iris.target, random_state=0)
        if self.mode == 'trainL':
            data = data[0:self.num1]
            targets = targets[0:self.num1]
        if self.mode == 'trainU':
            data = data[self.num1:self.num2]
            targets = targets[self.num1:self.num2]
        if self.mode == 'test':
            data = data[self.num2:]
            targets = targets[self.num2:]
        return data[item], targets[item]

    def __len__(self):
        if self.mode == 'trainL':
            return self.num1
        if self.mode == 'trainU':
            return self.num2 - self.num1
        if self.mode == 'test':
            return self.lenth-self.num2
