import argparse
from utils import *
import numpy as np
from sklearn.model_selection import KFold
from braindecode.torch_ext.util import set_random_seeds
from torch.utils.data import DataLoader, TensorDataset
import torch
import os


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='comparative runs')
    parser.add_argument('-dataset', type=str, default='sub54', help='dataset name')
    parser.add_argument('-model', type=str, default='EEGNetv4', help='model name')
    parser.add_argument('-fold', type=int, default=6, help='cross validation fold')
    parser.add_argument('-outrm', type=bool, default=True, help='outlier removal')
    parser.add_argument('-eps', type=float, default=7, help='eps for DBSCAN')
    parser.add_argument('-min_samples', type=int, default=5, help='min_samples for DBSCAN')
    parser.add_argument('-n_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-gpu', type=str, default='cuda:1', help='gpu device')


    args, _ = parser.parse_known_args()

    set_random_seeds(seed=15485485, cuda=True)

    train(args)


def train(args):

    data = load_data(args.dataset)
    order = subs_preorder()
    kf = KFold(n_splits=args.fold)
    model_path = './checkpoints/{}'.format(args.model)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    early_stopping = EarlyStopping(patience=20, verbose=True, path='{}/checkpoint.pth'.format(model_path))

    for idx, test_subj in enumerate(order):  # 54 subjects LOSO

        model = model_zoo(args.model, args.dataset)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        model.network.to(args.gpu)

        cv_set = np.array(order[idx+1:] + order[:idx])
        for cv_index, (train_index, valid_index) in enumerate(kf.split(cv_set)):

            x_train, y_train, x_valid, y_valid, x_test, y_test = data_split(train_index, valid_index, test_subj, data, cv_set)

            if args.outrm:
                x_train, y_train = clustering(x_train, y_train, args.eps, args.min_samples)
                x_valid, y_valid = clustering(x_valid, y_valid, args.eps, args.min_samples)

            x_train, y_train = np.concatenate(x_train, axis=0), np.concatenate(y_train, axis=0)
            x_valid, y_valid = np.concatenate(x_valid, axis=0), np.concatenate(y_valid, axis=0)

            train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train).float().unsqueeze(3), torch.from_numpy(y_train).long()), batch_size=64, shuffle=True)
            valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid).float().unsqueeze(3), torch.from_numpy(y_valid).long()), batch_size=64, shuffle=False)
            test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test).float().unsqueeze(3), torch.from_numpy(y_test).long()), batch_size=64, shuffle=False)

            for epoch in range(args.n_epochs):

                loss_train, acc_train = training_module(train_loader, model, optimizer, args.gpu, True)
                loss_valid, acc_valid = training_module(valid_loader, model, optimizer, args.gpu, False)
                print('Epoch: {:03d}, Train Loss: {:.5f}, Train Acc: {:.5f}, Valid Loss: {:.5f}, Valid Acc: {:.5f}'.format(epoch, loss_train, acc_train, loss_valid, acc_valid))

                early_stopping(loss_valid, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            _, acc_test = training_module(test_loader, model, optimizer, args.gpu, False)
            print('Test Acc: {:.5f}'.format(acc_test))

            if args.outrm:
                out_path = './results/{}_eps{}'.format(args.model, args.eps)
            else:
                out_path = './results/{}_eps-1'.format(args.model)
            with open(os.path.join(out_path, 'test_acc.txt'), 'a') as f:
                f.write('sub{}_fold{}: {}\n'.format(test_subj, cv_index, acc_test))
            f.close()


if __name__ == "__main__":

    main()

