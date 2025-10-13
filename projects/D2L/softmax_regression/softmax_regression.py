import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from torch.optim import SGD
from d2l import torch as d2l

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='data',
    train=True,
    transform=trans,
    download=True
)
mnist_test = torchvision.datasets.FashionMNIST(
    root='data',
    train=False,
    transform=trans,
    download=True
)

batch_size = 256
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=4)

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root='data',
        train=True,
        transform=trans,
        download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='data',
        train=False,
        transform=trans,
        download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=4))

train_iter, test_iter = load_data_fashion_mnist(batch_size)

# for X, y in train_iter:
#     print(f'X shape: {X.shape}, y shape: {y.shape}')
#     break

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

print(d2l.evaluate_accuracy(net, test_iter))