import numpy as np
import random
import pickle
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from os import listdir

from utils import set_random_seed, shuffle_in_union


def double_helix(size, min_z=-0.5 *np.pi, max_z=0.5 * np.pi, k=1.5):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_z, max_z)
        theta_neg = random.uniform(min_z, max_z)
        positions.append((np.cos(k * theta_pos), np.sin(k * theta_pos), theta_pos))
        original_positions.append(theta_pos)
        labels.append(1)
        positions.append((-np.cos(k * theta_neg), -np.sin(k * theta_neg), theta_neg))
        original_positions.append(theta_neg)
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_cosine(size, min_x=-0.6 * np.pi, max_x=0.6 * np.pi, k=1, delta=0.2):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_x, max_x)
        theta_neg = random.uniform(min_x, max_x)
        # positions.append((1, 1))
        positions.append((theta_pos, np.cos(k * theta_pos) + delta))
        original_positions.append(theta_pos)
        labels.append(1)
        # positions.append((0, 0))
        positions.append((theta_neg, np.cos(k * theta_neg) - delta))
        original_positions.append(theta_neg)
        labels.append(0)
    # add imbalance
    '''for index in range(size // 2):
        theta_pos = random.uniform(min_x, max_x)
        positions.append((theta_pos, np.cos(k * theta_pos) + delta))
        original_positions.append(theta_pos)
        labels.append(1)'''
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_fold_surface(size, min_x=-10, max_x=10, min_z=-1, max_z=1, k=1.2, bias=12, delta=1):
# def two_fold_surface(size, min_x=-10, max_x=10, min_z=-1, max_z=1, k=1.0, bias=10, delta=1):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 4):
        x_pos = random.uniform(min_x - delta/k, max_x + delta/k)
        x_neg = random.uniform(min_x + delta/k, max_x - delta/k)
        z_pos = random.uniform(min_z, max_z)
        z_neg = random.uniform(min_z, max_z)
        positions.append((x_pos, -k * np.abs(x_pos) + bias + delta, z_pos))
        original_positions.append((x_pos, -k, z_pos, max_x + delta))
        labels.append(1)
        positions.append((x_neg, -k * np.abs(x_neg) + bias - delta, z_neg))
        original_positions.append((x_neg, -k, z_neg, max_x - delta))
        labels.append(0)
    for index in range(size // 4):
        x_pos = random.uniform(min_x - delta / k, max_x + delta / k)
        x_neg = random.uniform(min_x + delta / k, max_x - delta / k)
        z_pos = random.uniform(min_z, max_z)
        z_neg = random.uniform(min_z, max_z)
        positions.append((x_pos, k * np.abs(x_pos) - bias - delta, z_pos))
        original_positions.append((x_pos, +k, z_pos, max_x + delta))
        labels.append(1)
        positions.append((x_neg, k * np.abs(x_neg) - bias + delta, z_neg))
        original_positions.append((x_neg, +k, z_neg, max_x - delta))
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def donut(size, min_theta=-np.pi, max_theta=np.pi, r=8, k=4):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_theta, max_theta)
        theta_neg = random.uniform(min_theta, max_theta)
        positions.append((r * np.cos(theta_pos) + np.sin(k * theta_pos) * np.cos(theta_pos),
                          r * np.sin(theta_pos) + np.sin(k * theta_pos) * np.sin(theta_pos),
                          np.cos(k * theta_pos)))
        original_positions.append(theta_pos)
        # original_positions.append((theta_pos, 1))
        labels.append(1)
        positions.append((r * np.cos(theta_neg) - np.sin(k * theta_neg) * np.cos(theta_neg),
                          r * np.sin(theta_neg) - np.sin(k * theta_neg) * np.sin(theta_neg),
                          - np.cos(k * theta_neg)))
        original_positions.append(theta_neg)
        # original_positions.append((theta_neg, -1))
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_sphere(size, min_theta=0, max_theta=1.0*np.pi, r_1=10, r_2=20):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 2):
        theta_pos = random.uniform(min_theta, max_theta)
        theta_neg = random.uniform(min_theta, max_theta)
        phi_pos = random.uniform(0, 2 * np.pi)
        phi_neg = random.uniform(0, 2 * np.pi)
        positions.append((r_1 * np.sin(theta_pos) * np.cos(phi_pos),
                          r_1 * np.sin(theta_pos) * np.sin(phi_pos),
                          r_1 * np.cos(theta_pos)))
        original_positions.append((np.sin(theta_pos) * np.cos(phi_pos), np.sin(theta_pos) * np.sin(phi_pos),
                                   np.cos(theta_pos)))
        labels.append(1)
        positions.append((r_2 * np.sin(theta_neg) * np.cos(phi_neg),
                          r_2 * np.sin(theta_neg) * np.sin(phi_neg),
                          r_2 * np.cos(theta_neg)))
        original_positions.append((np.sin(theta_neg) * np.cos(phi_neg), np.sin(theta_neg) * np.sin(phi_neg),
                                   np.cos(theta_neg)))
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def two_cycles(size, r1=10, r2=10, r3=50):
    positions = []
    original_positions = []
    labels = []
    for index in range(size // 4):
        pos_theta_one = random.uniform(-np.pi, np.pi)
        pos_theta_two = random.uniform(-np.pi, np.pi)
        neg_theta_one = random.uniform(-np.pi, np.pi)
        neg_theta_two = random.uniform(-np.pi, np.pi)

        positions.append((r1 * np.cos(pos_theta_one), r1 * np.sin(pos_theta_one)))
        # original_positions.append((r1 * np.cos(pos_theta_one), r1 * np.sin(pos_theta_one)))
        original_positions.append(pos_theta_one)
        labels.append(1)
        positions.append((r3 * np.cos(neg_theta_one), r3 * np.sin(neg_theta_one)))
        # original_positions.append((r3 * np.cos(neg_theta_one), r3 * np.sin(neg_theta_one)))
        original_positions.append(neg_theta_one)
        labels.append(0)

        positions.append((r2 * np.cos(pos_theta_two), r2 * np.sin(pos_theta_two)))
        # original_positions.append((r2 * np.cos(pos_theta_two), r2 * np.sin(pos_theta_two)))
        original_positions.append(pos_theta_two)
        labels.append(1)
        positions.append((r3 * np.cos(neg_theta_two), r3 * np.sin(neg_theta_two)))
        # original_positions.append((r3 * np.cos(neg_theta_two), r3 * np.sin(neg_theta_two)))
        original_positions.append(neg_theta_two)
        labels.append(0)
    labels = np.array(labels)
    labels = labels.reshape(len(labels), 1)
    positions = np.array(positions)
    original_positions = np.array(original_positions)
    return positions, labels, original_positions


def save_data_binary(config, dir_path):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    if config['dataset'] == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dir_path + '/data', train=False, download=True, transform=transform_test)
    else:
        trainset = torchvision.datasets.MNIST(root=dir_path + '/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=dir_path + '/data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config['test_batch_size'], shuffle=False)
    print('original train data size', len(trainloader))
    train_data_pos_one = []
    train_data_pos_two = []
    train_data_neg = []
    label_num = [0] * 3
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(inputs)
        if str(targets[0].item()) not in config['labels']['all']:
            continue
        if str(targets[0].item()) in config['labels']['neg']:
            if label_num[0] < config['sample_size']:
                label_num[0] += 1
                train_data_neg.append((inputs, torch.LongTensor([0])))
        else:
            if str(targets[0].item()) == config['labels']['pos'][0]:
                if label_num[1] < config['sample_size'] // 2:
                    train_data_pos_one.append((inputs, torch.LongTensor([1])))
                    label_num[1] += 1
            else:
                if label_num[2] < config['sample_size'] // 2:
                    train_data_pos_two.append((inputs, torch.LongTensor([1])))
                    label_num[2] += 1
    trainloader = train_data_pos_one, train_data_pos_two, train_data_neg
    test_data_pos_one = []
    test_data_pos_two = []
    test_data_neg = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if str(targets[0].item()) not in config['labels']['all']:
            continue
        if str(targets[0].item()) in config['labels']['neg']:
            test_data_neg.append((inputs, torch.LongTensor([0])))
        else:
            if str(targets[0].item()) == config['labels']['pos'][0]:
                test_data_pos_one.append((inputs, torch.LongTensor([1])))
            else:
                test_data_pos_two.append((inputs, torch.LongTensor([1])))
    testloader = test_data_pos_one, test_data_pos_two, test_data_neg
    print('train data pos one size', len(train_data_pos_one))
    print('train data pos two size', len(train_data_pos_two))
    print('train data neg', len(train_data_neg))
    print('test data pos one size', len(test_data_pos_one))
    print('test data pos two size', len(test_data_pos_two))
    print('test data neg', len(test_data_neg))
    print(label_num)
    if config['dataset'] == 'CIFAR10':
        pickle_out_train = open(dir_path + "/data/train_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                                "wb")
        pickle_out_test = open(dir_path + "/data/test_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "wb")
    else:
        pickle_out_train = open(dir_path + "/data/train_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                                "wb")
        pickle_out_test = open(dir_path + "/data/test_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "wb")
    pickle.dump(trainloader, pickle_out_train)
    pickle_out_train.close()
    pickle.dump(testloader, pickle_out_test)
    pickle_out_test.close()


def load_data_binary_from_pickle(config, dir_path):
    if config['dataset'] == 'CIFAR10':
        pickle_in_train = open(dir_path + "/data/train_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "rb")
        pickle_in_test = open(dir_path + "/data/test_cifar10_binary_" + "_".join(config['labels']['all']) + ".pickle",
                              "rb")
    else:
        pickle_in_train = open(dir_path + "/data/train_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                               "rb")
        pickle_in_test = open(dir_path + "/data/test_mnist_binary_" + "_".join(config['labels']['all']) + ".pickle",
                              "rb")
    trainloader = pickle.load(pickle_in_train)
    pickle_in_train.close()
    testloader = pickle.load(pickle_in_test)
    pickle_in_test.close()
    return trainloader, testloader

