import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--algorithm', type=str, choices=['svm'], default='svm',
                    help='The algorithm used to predict. Default: svm')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='Number of training epoch. Default: 20')
parser.add_argument('--batch_size', type=int, default=32,
                    help='The number of batch_size. Default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='Learning rate during optimization. Default: 1e-3')
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
