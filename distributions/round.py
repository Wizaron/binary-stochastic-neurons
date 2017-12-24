from numbers import Number

import torch
from torch.autograd import Variable
from distribution import Distribution
from utils import broadcast_all

class Round(Distribution):
    has_enumerate_support = True

    def __init__(self, probs):
        self.probs, = broadcast_all(probs)
        if isinstance(probs, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.probs.size()
        super(Round, self).__init__(batch_shape)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return torch.round(self.probs.expand(shape))

    def log_prob(self, value):
        self._validate_log_prob_arg(value)
        param_shape = value.size()
        probs = self.probs.expand(param_shape)
        # compute the log probabilities for 0 and 1
        log_pmf = (torch.stack([1 - probs, probs], dim=-1)).log()
        # evaluate using the values
        return log_pmf.gather(-1, value.unsqueeze(-1).long()).squeeze(-1)

    def entropy(self):
        p = torch.stack([self.probs, 1.0 - self.probs])
        p_log_p = torch.log(p) * p
        p_log_p[p == 0] = 0
        return -p_log_p.sum(0)

    def enumerate_support(self):
        values = torch.arange(2).long()
        values = values.view((-1,) + (1,) * len(self._batch_shape))
        values = values.expand((-1,) + self._batch_shape)
        if self.probs.is_cuda:
            values = values.cuda(self.probs.get_device())
        if isinstance(self.probs, Variable):
            values = Variable(values)
        return values
