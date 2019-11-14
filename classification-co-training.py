from random import shuffle

from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
from torch.autograd import variable
from torch import optim
from torch.utils.data import DataLoader
from advertorch.attacks import GradientSignAttack
import argparse
import numpy as np
import os
import math
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from models.iris_classification_model import iris_classifier
from dataset.dataclass import irisdata
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Deep Co-Training for Image Classification')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--batchsize', '-b', default=10, type=int)
parser.add_argument('--lambda_cot_max', default=10, type=int)
parser.add_argument('--lambda_diff_max', default=0.5, type=float)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--warm_up', default=80.0, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=1e-4, type=float)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--num_class', default=3, type=int)
parser.add_argument('--iris_dir', default='./data', type=str)
parser.add_argument('--svhn_dir', default='./data', type=str)
parser.add_argument('--tensorboard_dir', default='tensorboard/', type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--dataset', default='iris', type=str, help='choose svhn or iris, svhn is not implemented yey')
parser.add_argument('--alpha',type=float, default=1.0, help="alpha for jsd")
args = parser.parse_args()
args.tensorboard_dir = args.tensorboard_dir + str(args.alpha) +"/" +str(args.seed)
args.checkpoint_dir = args.tensorboard_dir

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

if not os.path.isdir(args.tensorboard_dir):
    from pathlib import Path
    Path(args.tensorboard_dir).mkdir(parents=True, exist_ok=False)

writer = SummaryWriter(args.tensorboard_dir)
start_epoch = 0
end_epoch =args.epochs
class_num = args.num_class
batch_size = args.batchsize

if args.dataset == 'iris':
    U_batch_size = int(batch_size * 2./3)
    S_batch_size = batch_size-U_batch_size
else:
    U_batch_size = int(batch_size * 72 / 73)
    S_batch_size = batch_size - U_batch_size

lambda_cot_max = args.lambda_cot_max
lambda_diff_max = args.lambda_diff_max
lambda_cot = 0.0
lambda_diff = 0.0
best_acc = 0.0


def adjust_learning_rate(optimizer, epoch):
    epoch = epoch +1
    lr = args.base_lr * (1.0 + math.cos((epoch - 1) * math.pi / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lamda(epoch):
    epoch = epoch + 1
    global lambda_cot
    global lambda_diff
    if epoch <= args.warm_up:
        lambda_cot = lambda_cot_max*math.exp(-5*(1-epoch/args.warm_up)**2)
        lambda_diff = lambda_diff_max*math.exp(-5*(1-epoch/args.warm_up)**2)
    else:
        lambda_cot = lambda_cot_max
        lambda_diff = lambda_diff_max


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    ce = nn.CrossEntropyLoss()
    loss1 = ce(logit_S1, labels_S1)
    loss2 = ce(logit_S2, labels_S2)
    return (loss1+loss2)


def loss_cot(U_p1, U_p2, alpha):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a1 = 0.5 * (S(U_p1) + S(U_p2))
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)
    loss2 = S(U_p1) * LS(U_p1)
    loss2 = -torch.sum(loss2)
    loss3 = S(U_p2) * LS(U_p2)
    loss3 = -torch.sum(loss3)
    return (loss1 - 0.5 * alpha * (loss2 + loss3))/U_batch_size


trainsetL = irisdata(20, 120, mode='trainL')
trainsetU = irisdata(20, 120, mode='trainU')
testset = irisdata(20, 120, mode='test')

start_epoch=0
net1 = iris_classifier()
net2 = iris_classifier()


S_loader1 = DataLoader(trainsetL, batch_size=S_batch_size)
S_loader2 = DataLoader(trainsetL, batch_size=S_batch_size)
U_loader = DataLoader(trainsetU, batch_size=U_batch_size)
testloader = DataLoader(testset, batch_size=args.batchsize)
adversary1 = GradientSignAttack(
    net1, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon)

adversary2 = GradientSignAttack(
    net2, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=args.epsilon)

if args.dataset == 'iris':
    step = int(len(trainsetL)/batch_size)

params = list(net1.parameters())+list(net2.parameters())
optimizer = optim.SGD(params, lr=args.base_lr, momentum=0.99)


for epoch in range(start_epoch, end_epoch):
      S_iter1 = iter(S_loader1)
      S_iter2 = iter(S_loader2)
      U_iter = iter(U_loader)
      running_loss = 0.0
      ls = 0.0
      lc = 0.0
      print('epoch=', epoch+1)
      for i in tqdm(range(step)):
          inputs_S1, labels_S1 = S_iter1.next()
          inputs_S2, labels_S2 = S_iter2.next()
          inputs_U, labels_U = U_iter.next()  # not use the labels_U

          logit_S1 = net1(inputs_S1)
          logit_S2 = net2(inputs_S2)
          logit_U1 = net1(inputs_U)
          logit_U2 = net2(inputs_U)

          optimizer.zero_grad()
          net1.zero_grad()
          net2.zero_grad()

          Loss_sup = loss_sup(logit_S1, logit_S2, labels_S1, labels_S2)
          Loss_cot = loss_cot(logit_U1, logit_U2, args.alpha)

          total_loss = Loss_sup + Loss_cot
          total_loss.backward()
          optimizer.step()

          running_loss += total_loss.item()
          ls += Loss_sup.item()
          lc += Loss_cot.item()
          writer.add_scalar('data/loss', {'loss_sup:', Loss_sup.item(), 'loss_cot:', Loss_cot.item(), 'total_loss:',total_loss.item()}, (epoch)*(step)+i)
      print('total_loss = ', total_loss)
      print('\n')

torch.save(net1, 'net1.pkl')
torch.save(net2, 'net2.pkl')

for batch_idx, (inputs, targets) in enumerate(testloader):
    total1 = 0.0
    correct1 = 0.
    total2 = 0.0
    correct2 = 0.0
    outputs1 = net1(inputs)
    predicted1 = outputs1.max(1)
    total1 += targets.size(0)
    correct1 += predicted1[1].eq(targets).sum().item()

    outputs2 = net2(inputs)
    predicted2 = outputs2.max(1)
    total2 += targets.size(0)
    correct2 += predicted2[1].eq(targets).sum().item()
    print('\n net1 test acc: %.3f%% (%d%d) | net2 test acc: %.3f%%(%d/%d)', (100.*correct1/total1, (correct1, total1), 100.*correct2/total2, (correct2, total2)))
    writer.add_scalars('data/testing_accuracy',
                   {'net1 acc': 100. * correct1 / total1, 'net2 acc': 100. * correct2 / total2}, epoch)


