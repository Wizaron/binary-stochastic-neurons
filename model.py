import torch.nn as nn
import torch.nn.functional as F

from activations import DeterministicBinaryActivation, StochasticBinaryActivation
from utils import Hardsigmoid

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.mode = None
        self.estimator = None

class NonBinaryNet(Net):

    def __init__(self):
        super(NonBinaryNet, self).__init__()

        self.fc1 = nn.Linear(784, 100)
        self.act = Hardsigmoid()
        self.fc2 = nn.Linear(100, 10)

    def forward(self, input):
        x, slope = input
        x = x.view(-1, 784)
        x_fc1 = self.act(slope * self.fc1(x))
        x_fc2 = self.fc2(x_fc1)
        x_out = F.log_softmax(x_fc2, dim=1)
        return x_out

class BinaryNet(Net):

    def __init__(self, mode='Deterministic', estimator='ST'):
        super(BinaryNet, self).__init__()

        assert mode in ['Deterministic', 'Stochastic']
        assert estimator in ['ST', 'REINFORCE']
        #if mode == 'Deterministic':
        #    assert estimator == 'ST'

        self.mode = mode
        self.estimator = estimator

        self.fc1 = nn.Linear(784, 100)
        if self.mode == 'Deterministic':
            self.act = DeterministicBinaryActivation(estimator=estimator)
        elif self.mode == 'Stochastic':
            self.act = StochasticBinaryActivation(estimator=estimator)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, input):
        x, slope = input
        x = x.view(-1, 784)
        x_fc1 = self.act((self.fc1(x), slope))
        x_fc2 = self.fc2(x_fc1)
        x_out = F.log_softmax(x_fc2, dim=1)
        return x_out
