from braindecode.models import Deep4Net, EEGNetv4, EEGInception, EEGResNet, EEGITNet, TIDNet
# from braindecode.models.deep4 import Deep4Net
# from braindecode.models.eegnet import EEGNetv4
import torch.nn.functional as F
import torch
import h5py
from os.path import join as pjoin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def model_zoo(name, dataset):
    """Return a list of model names in the model zoo."""

    if dataset == 'sub54':
        channels = 62
        timepoints = 1000
        num_classes = 2
    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))

    if name == 'Deep4Net':
        model = Deep4Net(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_conv_length='auto')
    elif name == 'EEGNetv4':
        model = EEGNetv4(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints, final_conv_length='auto')
    elif name == 'EEGInception':
        model = EEGInception(in_channels=channels, n_classes=num_classes, input_window_samples=timepoints)
    elif name == 'EEGResNet':
        model = EEGResNet(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints)
    elif name == 'EEGITNet':
        model = EEGITNet(in_channels=channels, n_classes=num_classes, input_window_samples=timepoints)
    elif name == 'TIDNet':
        model = TIDNet(in_chans=channels, n_classes=num_classes, input_window_samples=timepoints)
    else:
        raise ValueError('Invalid model name: {}'.format(name))

    return model


def load_data(dataset):
    """Load dataset."""

    if dataset == 'sub54':
        path = './data/KU_mi_smt.h5'
        data = h5py.File(path, 'r')

    else:
        raise ValueError('Invalid dataset name: {}'.format(dataset))

    return data


def data_split(train_index, valid_index, test_subj, data, cv_set):
    """Split the dataset into training, validation and test sets."""

    train_subjs = cv_set[train_index]
    valid_subjs = cv_set[valid_index]
    X_train, Y_train = get_multi_data(data, train_subjs)
    X_val, Y_val = get_multi_data(data, valid_subjs)
    X_test, Y_test = get_data(data, test_subj)
    X_test, Y_test = X_test[:], Y_test[:]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def get_data(dfile, subj):

    dpath = '/s' + str(subj)
    X = dfile[pjoin(dpath, 'X')]
    Y = dfile[pjoin(dpath, 'Y')]

    return X, Y


def get_multi_data(dfile, subjs):

    Xs = []
    Ys = []
    for s in subjs:
        x, y = get_data(dfile, s)
        Xs.append(x[:])
        Ys.append(y[:])

    return Xs, Ys


def dbscan(data, eps, min_samples):
    """DBSCAN clustering with PCA"""

    data = StandardScaler().fit_transform(np.mean(data, axis=1))
    pca = PCA(n_components=10)
    data = pca.fit_transform(data)
    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # print how many clusters are there in total and how many are noise points
    labels = db.labels_

    return db, labels


def clustering(X, Y, eps, min_samples):

    list_x, list_y = [], []

    for _, (x, y) in enumerate(zip(X, Y)):

        idx = np.argsort(y)
        try:
            x, y = x[:][idx], y[:][idx]
        except IndexError:
            pass

        # clustering in each class
        class1, class2 = x[:200], x[200:]
        dbs1, label1 = dbscan(class1, eps=eps, min_samples=min_samples)
        dbs2, label2 = dbscan(class2, eps=eps, min_samples=min_samples)

        class1, class2 = class1[label1 != -1], class2[label2 != -1]
        label1, label2 = np.zeros((len(label1[label1 != -1])), dtype=np.int64), \
                         np.ones((len(label2[label2 != -1])), dtype=np.int64)

        # concatenate
        list_x.append(class1), list_x.append(class2), list_y.append(label1), list_y.append(label2)

    return list_x, list_y


def accuracy(pred, label):

    count = 0
    for i in range(label.shape[0]):
        try:
            if np.argmax(pred[i]) == label[i]:
                count += 1
        except RuntimeError:
            if np.argmax(pred[i]) == np.argmax(label[i]):
                count += 1

    return count / label.shape[0]


def subs_preorder():

    return [35, 47, 46, 37, 13, 27, 12, 32, 53, 54, 4, 40, 19, 41, 18, 42, 34, 7, 49, 9, 5, 48, 29, 15, 21, 17, 31, 45,
            1, 38, 51, 8, 11, 16, 28, 44, 24, 52, 3, 26, 39, 50, 6, 23, 2, 14, 25, 20, 10, 33, 22, 43, 36, 30]


def training_module(loader, model, optimizer, device, train_mode, mixup=False):

    if train_mode:
        model.train()
        # for param in model.parameters():
        #     param.requires_grad = True
    else:
        model.eval()
        # for param in model.parameters():
        #     param.requires_grad = False

    loss_all, acc_all = 0, 0
    for batch_idx, (data, label) in enumerate(loader):

        data, label = data.to(device), label.to(device)
        if mixup and train_mode:
            data, label_a, label_b, lam = mixup_data(data, label, 1.0, True)
            output = model(data)
            loss = mixup_criterion(F.nll_loss, output, label_a, label_b, lam)
        else:
            output = model(data)
            loss = F.nll_loss(output, label)
        loss_all += loss.item()
        acc_all += accuracy(output.detach().cpu().numpy(), label)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_all / len(loader), acc_all / len(loader)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """Returns mixed inputs, pairs of targets, and lambda"""

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # model.save_networks(self.name)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
