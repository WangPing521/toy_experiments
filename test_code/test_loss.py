import torch
from deepclustering2.loss import Entropy, JSD_div, KL_div
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd


#
from architectures.enet import Enet
from loss_functions.crossentropy import SimplexCrossEntropyLoss


def plot_distribution(prob: torch.Tensor, figure_num=1) -> None:
    plt.figure(figure_num)
    prob_np = prob.data.cpu().numpy().ravel()
    plt.clf()
    pd.Series(prob_np).plot.density()
    plt.xlim([-0.1, 1.1])
    plt.show(block=False)
    plt.pause(0.001)


def plot_pred(prod, figure_num=1) -> None:
    plt.figure(figure_num)
    plt.clf()
    plt.imshow(prod[0][1].data.cpu().numpy().squeeze())
    plt.colorbar()
    plt.show(block=False)
    plt.pause(0.001)


def test_jsd_loss():
    imgs = torch.randn(1, 1, 16, 16).cuda()
    net1 = Enet(input_dim=1, num_classes=2)
    net1 = net1.cuda()
    net2 = Enet(input_dim=1, num_classes=2)
    net2 = net2.cuda()

    optim1 = torch.optim.Adam(net1.parameters())
    optim2 = torch.optim.Adam(net2.parameters())
    criterion = JSD_div()
    criterion2 = Entropy()
    fig = plt.figure()

    mask = torch.zeros(imgs.shape[0], imgs.shape[2], imgs.shape[3])
    mask = mask.cuda()
    mask[:, :, 8:] = 1

    for i in range(10000):
        optim1.zero_grad()
        optim2.zero_grad()
        pred_logit = net1(imgs)
        loss1 = criterion2(F.softmax(pred_logit, 1)).mean()

        pred_logits1 = net1(imgs)
        pred_logits2 = net2(imgs)
        loss2 = criterion([F.softmax(pred_logits1, 1), F.softmax(pred_logits2, 1)])
        assert loss2.shape == mask.shape
        loss2_ = loss2 * mask
        loss2_ = loss2_.sum() / mask.sum()
        loss = loss1 + 1 * loss2_
        loss.backward()

        optim1.step()
        optim2.step()

        if i % 10 == 0:
            # print(F.softmax(pred_logits, 1).data)
            # plot_distribution(F.softmax(pred_logits1, 1), 1)
            # plot_distribution(F.softmax(pred_logits2, 1), 2)
            plot_pred(F.softmax(pred_logits1, 1), 1)
            plot_pred(F.softmax(pred_logits2, 1), 2)
            plt.figure(3)
            plt.clf()
            plt.imshow(loss2.data.cpu().numpy().squeeze())
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.0001)
            print(f'entropy loss:{loss1.item()}, JSD loss:{loss2_.item()}')


def test_kl_loss():
    imgs = torch.randn(1, 1, 16, 16).cuda()
    net1 = Enet(input_dim=1, num_classes=2)
    net1 = net1.cuda()
    net2 = Enet(input_dim=1, num_classes=2)
    net2 = net2.cuda()

    optim1 = torch.optim.Adam(net1.parameters())
    optim2 = torch.optim.Adam(net2.parameters())
    criterion = Entropy()
    criterion2 = KL_div()
    fig = plt.figure()

    mask = torch.zeros(imgs.shape[0], imgs.shape[2], imgs.shape[3])
    mask = mask.cuda()
    mask[:, :, 8:] = 1

    for i in range(10000):
        optim1.zero_grad()
        optim2.zero_grad()
        pred_logit = net1(imgs)
        loss1 = criterion(F.softmax(pred_logit, 1)).mean()

        pred_logits1 = net1(imgs)
        pred_logits2 = net2(imgs)
        loss2 = criterion2(F.softmax(pred_logits2, 1), F.softmax(pred_logits1, 1))
        assert loss2.shape == mask.shape
        loss2_ = loss2 * mask
        loss2_ = loss2_.sum() / mask.sum()
        loss = loss1 + 0.001 * loss2_
        loss.backward()

        optim1.step()
        optim2.step()

        if i % 10 == 0:
            plot_pred(F.softmax(pred_logits1, 1), 1)
            plot_pred(F.softmax(pred_logits2, 1), 2)
            plt.figure(3)
            plt.clf()
            plt.imshow(loss2.data.cpu().numpy()[0].squeeze())
            plt.colorbar()
            plt.show(block=False)
            plt.pause(0.0001)
            print(f'entropy loss:{loss1.item()}, JSD loss:{loss2_.item()}')


def test_ignore_index():

    imgs = torch.randn(1, 1, 16, 16)
    net1 = Enet(input_dim=1, num_classes=2)
    gt = torch.randint(0,3,(1,16,16))
    gt[gt==2]=255
    criterion = SimplexCrossEntropyLoss(ignore_index=255)
    loss = criterion(net1(imgs), gt.long())


if __name__ == '__main__':
    # test_jsd_loss()
    # test_kl_loss()
    test_ignore_index()