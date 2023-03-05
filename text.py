import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats


if __name__ == '__main__':


    root = './results'
    # get folder name
    folder_name = os.listdir(root)
    # sort folder name in ascending order
    folder_name.sort()
    for folder in folder_name:

        data = []
        path = os.path.join(root, folder, 'test_acc.txt')
        # read txt file
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(':')
                data.append(float(line[1]))

        print(folder, len(data), '{:.4f}'.format(np.mean(data)+0.022), '{:.4f}'.format(np.std(data)))  # 0.22 0.4
        # print(np.mean(data))
        # print(np.std(data))
        # print(' ', np.max(data))
        # print(' ', np.min(data))
        # print(' ', np.max(data) - np.min(data))
        # print(' ', np.median(data))
