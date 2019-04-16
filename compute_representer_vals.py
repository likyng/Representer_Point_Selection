#!/usr/bin/env python
# coding: utf-8
import time
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import pickle
from scipy.stats.stats import pearsonr
dtype = torch.cuda.FloatTensor


class softmax(torch.nn.Module):
    def __init__(self, W):
        super(softmax, self).__init__()
        self.W = Variable(torch.from_numpy(W).type(dtype), requires_grad=True)

    def forward(self, x, y):
        """Calculate loss for the loss function and L2 regularizer

        Arguments:
            x: input data
            y: labels

        Returns:
            Phi: torch tensor, model predictions shape(1) [TODO: check!]
            L2: torch tensor, L2 distance shape(1)"""
        D = (torch.matmul(x, self.W))
        D_max, _ = torch.max(D, dim=1, keepdim=True)
        D = D - D_max
        A = torch.log(torch.sum(torch.exp(D), dim=1))
        B = torch.sum(D * y, dim=1)
        Phi = torch.sum(A - B)
        W1 = torch.squeeze(self.W)
        L2 = torch.sum(torch.mul(W1, W1))
        return (Phi, L2)


def softmax_np(x):
    """Returns the softmax

    Arguments:
        x: torch tensor

    Returns:
        softmax: numpy tensor"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def load_data(dataset):
    """Loads the specified dataset and returns the required values.

    Arguments:
        dataset: string, name of the dataset to be loaded. Either Cifar or AwA.

    Returns:
        x_val: np array, shape(#train samples x 4097)
        y_val: np array, shape(#train samples x #classes)
        model: torch model, contains the DNN model with
            model.W.shape = (4097 x #classes), from W_36"""
    if dataset == "Cifar":
        with open("data/weight_323436.pkl", "rb") as input_file:
            [
                W_32, W_34, W_36, intermediate_output_32,
                intermediate_output_34, intermediate_output_36
            ] = pickle.load(input_file, encoding='latin1')

            print(
                (softmax_np(
                    np.matmul(
                        np.concatenate(
                            [
                                intermediate_output_34, np.ones((
                                    intermediate_output_34.shape[0], 1))],
                            axis=1),
                        W_36)) - intermediate_output_36)[:5, :])

            print(intermediate_output_36[:5, :])
        print('done loading')
        model = softmax(W_36)
        model.cuda()
        x_val = np.concatenate(
            [
                intermediate_output_34,
                np.ones((intermediate_output_34.shape[0], 1))], axis=1)
        y_val = intermediate_output_36
        return (x_val, y_val, model)

    elif dataset == "AwA":
        with open("data/weight_bias.pickle", "rb") as input_file:
            [weight, bias] = pickle.load(input_file, encoding='latin1')
        train_feature = np.squeeze(np.load('data/train_feature_awa.npy'))
        train_output = np.squeeze(np.load('data/train_output_awa.npy'))
        weight = np.transpose(
            np.concatenate([weight, np.expand_dims(bias, 1)], axis=1))
        train_feature = np.concatenate(
            [train_feature, np.ones((train_feature.shape[0], 1))], axis=1)
        train_output = softmax_np(train_output)
        model = softmax(weight)
        model.cuda()
        return (train_feature, train_output, model)


def to_np(x):
    return x.data.cpu().numpy()


def backtracking_line_search(optimizer, model, grad, x, y, loss, N, lmbd,
                             beta=0.5):
    """Implements the backtracking line search.

    Arguments:
        optimizer: torch optimizer
        model: torch model
        grad: gradients
        x: input, e.g. layer weights previous to y
        y: output, e.g. layer weights posterior to x
        loss: torch tensor w/ one float, loss
        N: int, train dataset size (length of y)
        lmbd: float, lambda of the L2 reguliser
        beta: float, search control parameter tao of the line search gd
    """
    t = 10.0    # step size alpha_0
    W_O = to_np(model.W)
    grad_np = to_np(grad)

    while(True):
        # Update the model's weights
        model.W = Variable(
            torch.from_numpy(W_O - t * grad_np).type(dtype), requires_grad=True)
        val_n = 0.0
        # Does one forward prop. on layer 34th weights (x) and
        # layer 36 output (y). According to formular 3 (p. 4) in the paper.
        (Phi, L2) = model(x, y)
        val_n = Phi / N + L2 * lmbd
        if t < 0.0000000001:
            print("t too small")
            break
        # Testing for Armijo-Goldstein condition, if not satisfied decrease t
        if to_np(val_n - loss + t * torch.norm(grad) ** 2 / 2) >= 0:
            t = beta * t
        # Armijo-Goldstein condition reached --> finish
        else:
            break


def softmax_torch(temp, N):
    """Calculation for softmax in torch, which avoids numerical overflow

    Arguments:
        temp: torch tensor, shape(#training dataset x #classes)
        N: int, training dataset size

    Returns:
        """
    max_value, _ = torch.max(temp, 1, keepdim=True)
    temp = temp - max_value
    D_exp = torch.exp(temp)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N, 1)
    return D_exp.div(D_exp_sum.expand_as(D_exp))


def train(X, Y, model, args):
    x = Variable(torch.FloatTensor(X).cuda())
    y = Variable(torch.FloatTensor(Y).cuda())
    N = len(Y)
    min_loss = 10000.0
    optimizer = optim.SGD([model.W], lr=1.0)
    for epoch in range(args.epoch):
        phi_loss = 0
        optimizer.zero_grad()
        (Phi, L2) = model(x, y)
        loss = L2 * args.lmbd + Phi / N
        phi_loss += to_np(Phi / N)
        loss.backward()
        temp_W = model.W.data
        grad_loss = to_np(torch.mean(torch.abs(model.W.grad)))
        # save the W with lowest loss
        if grad_loss < min_loss:
            if epoch == 0:
                init_grad = grad_loss
            min_loss = grad_loss
            best_W = temp_W
            if min_loss < init_grad / 200:
                print('stopping criteria reached in epoch :{}'.format(epoch))
                break

        # The model weights are updated in this step
        backtracking_line_search(
            optimizer, model, model.W.grad, x, y, loss, N, args.lmbd, 0.5)

        if epoch % 100 == 0:
            print('Epoch:{:4d}\tloss:{}\tphi_loss:{}\tgrad:{}'.format(
                epoch, to_np(loss), phi_loss, grad_loss))

    # caluculate W based on the representer theorem's decomposition
    temp = torch.matmul(x, Variable(best_W))
    softmax_value = softmax_torch(temp, N)

    # derivative of softmax cross entropy
    weight_matrix = softmax_value - y
    weight_matrix = torch.div(weight_matrix, (-2.0 * args.lmbd * N))
    print(weight_matrix[:5, :5].cpu())
    w = torch.matmul(torch.t(x), weight_matrix)
    print(w[:5, :5].cpu())

    # calculate y_p, which is the prediction based on decomposition of W
    # by representer theorem
    temp = torch.matmul(x, w.cuda())
    print(temp[:5, :5].cpu())
    softmax_value = softmax_torch(temp, N)
    y_p = to_np(softmax_value)
    print(y_p[:5, :])

    print('L1 difference between ground truth prediction and prediction by',
          ' representer theorem decomposition')
    print(np.mean(np.abs(to_np(y) - y_p)))

    print('Pearson correlation between ground truth prediction and',
          ' prediciton by representer theorem')
    y = to_np(y)
    corr, _ = (pearsonr(y.flatten(), (y_p).flatten()))
    print(corr)
    sys.stdout.flush()
    return to_np(weight_matrix)


def main(args):
    x, y, model = load_data(args.dataset)
    start = time.time()
    weight_matrix = train(x, y, model, args)
    end = time.time()
    print('Computational time')
    print(end - start)

    np.savez(
        "output/weight_matrix_{}".format(args.dataset),
        weight_matrix=weight_matrix)
    with open("output/weight_matrix_{}.pkl".format(args.dataset), "wb") as output_file:
        pickle.dump(
            [weight_matrix, y], output_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbd', type=float, default=0.003)
    parser.add_argument('--epoch', type=int, default=3000)
    parser.add_argument('--dataset', type=str, default="Cifar")
    args = parser.parse_args()
    print(args)
    main(args)
