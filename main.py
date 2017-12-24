import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from model import NonBinaryNet, BinaryNet

# Training settings
parser = argparse.ArgumentParser(description='Binary Neurons')
parser.add_argument('--binary', action='store_true', default=False,
                    help='Use binary activations instead of float')
parser.add_argument('--stochastic', action='store_true', default=False,
                    help='Use stochastic activations instead of deterministic [active iff `--binary`]')
parser.add_argument('--reinforce', action='store_true', default=False,
                    help='Use REINFORCE Estimator instead of Straight Through Estimator [active iff `--binary`]')
parser.add_argument('--slope-annealing', action='store_true', default=False,
                    help='Use slope annealing trick')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

# Model, activation type, estimator type
if args.binary:
    if args.stochastic:
        mode = 'Stochastic'
    else:
        mode = 'Deterministic'
    if args.reinforce:
        estimator = 'REINFORCE'
    else:
        estimator = 'ST'
    model = BinaryNet(mode=mode, estimator=estimator)
else:
    model = NonBinaryNet()
    mode = None
    estimator = None

# Cuda
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    model.cuda()

# Dataset
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Slope annealing
if args.slope_annealing:
    get_slope = lambda epoch : 1.0 * (1.005 ** (epoch - 1))
else:
    get_slope = lambda epoch : 1.0

# Training procedure
def train(epoch):

    slope = get_slope(epoch)

    print '# Epoch : {} - Slope : {}'.format(epoch, slope)

    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model((data, slope))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss /= len(train_loader)
    train_loss = train_loss.data[0]

    print 'Training Loss : {}'.format(train_loss)

    return train_loss

# Testing procedure
def test(epoch, best_acc):
    slope = get_slope(epoch)

    model.eval()
    test_loss = 0.0
    correct = 0.0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model((data, slope))
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    print 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, int(correct), len(test_loader.dataset),
          100. * test_acc)

    if test_acc >= best_acc:
        torch.save(model.state_dict(), './models/{}.pth'.format(model_name))

    return test_loss, test_acc

model_name = '{}-{}-{}-{}'.format(model.__class__.__name__, model.mode, model.estimator, args.slope_annealing)
print 'Model : {}'.format(model_name.replace('-', ' - '))

best_acc = 0.0
log_file = open('./logs/{}.log'.format(model_name), 'w')
log_file.write('Epoch,TrainLoss,TestLoss,TestAccuracy\n')
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    test_loss, test_acc = test(epoch, best_acc)
    log_file.write('{},{},{},{}\n'.format(epoch, train_loss, test_loss, test_acc))
    log_file.flush()
log_file.close()
