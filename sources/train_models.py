import torch
import matplotlib.pyplot as plt
import numpy as np
import sys


from utils import set_random_seed, shuffle_in_union
from data import save_data_binary, load_data_binary_from_pickle
from elastic_effect import build_model, simple_train_batch, simple_test_batch
from clustering import clustering_images

torch.set_default_tensor_type('torch.cuda.FloatTensor')
plt.switch_backend('agg')


def experiments(seed, config, dir_path):
    print('seed', seed)
    set_random_seed(seed)
    if 'Prob' in config['model'] and config['dataset'] == 'MNIST':
        model_path = dir_path + '/models/' + config['dataset'] + '_' + "_".join(config['labels']['all']) + '_' + \
                    str(seed) + '_' + config['model'] + '_50epoch.pt'
    else:
        model_path = dir_path + '/models/' + config['dataset'] + '_' + "_".join(config['labels']['all']) + '_' + \
                    str(seed) + '_' + config['model'] + '.pt'
    print('load data from pickle')
    trainloader, testloader = load_data_binary_from_pickle(config, dir_path)
    train_data_pos_one, train_data_pos_two, train_data_neg = trainloader
    train_data = []
    train_data.extend(train_data_pos_one)
    train_data.extend(train_data_pos_two)
    train_data.extend(train_data_neg)
    train_data_label = []
    train_data_label.extend([1] * len(train_data_pos_one))
    train_data_label.extend([2] * len(train_data_pos_two))
    train_data_label.extend([0] * len(train_data_neg))
    train_data_index_list = np.arange(len(train_data))
    train_data_label = np.array(train_data_label)
    train_data_index_list, train_data_label = shuffle_in_union(train_data_index_list, train_data_label)
    new_train_data = []
    for idx in range(len(train_data_index_list)):
        index = train_data_index_list[idx]
        new_train_data.append(train_data[index])
    train_data = new_train_data
    index_list = []
    for idx in range(len(train_data)):
        if train_data_label[idx] > 0:
            index_list.append(idx)
    print('clustering images')
    clustering_images(train_data, index_list, train_data_label[index_list])
    test_data_pos_one, test_data_pos_two, test_data_neg = testloader
    test_data = []
    test_data.extend(test_data_pos_one)
    test_data.extend(test_data_pos_two)
    test_data.extend(test_data_neg)
    np.random.shuffle(test_data)
    print('build model')
    model, loss_function, optimizer = build_model(config)
    print('train model')
    simple_train_batch(train_data, model, loss_function, optimizer, config)
    print('save model')
    torch.save(model.state_dict(), model_path)
    print('load model')
    model.load_state_dict(torch.load(model_path))
    train_res, train_np = simple_test_batch(train_data, model, config)
    test_res, test_np = simple_test_batch(test_data, model, config)
    print('train accuracy', train_res)
    print('test accuracy', test_res)


def run_experiments():
    set_random_seed(666)
    dataset = sys.argv[1].split('=')[1]
    model_option = sys.argv[2].split('=')[1]
    loss_option = sys.argv[3].split('=')[1]
    pos_one = sys.argv[4].split('=')[1]
    pos_two = sys.argv[5].split('=')[1]
    neg_one = sys.argv[6].split('=')[1]
    neg_two = sys.argv[7].split('=')[1]
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    dir_path = '/path/to/experiments/dir'
    config = {'batch_size': 1, 'epoch_num': 200, 'lr': 3e-3, 'test_batch_size': 1,
              'sample_size': 1000, 'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'labels': {'pos': [pos_one, pos_two], 'neg': [neg_one, neg_two],
                         'all': [pos_one, pos_two, neg_one, neg_two]}, 'dataset': dataset,
              'model': model_option, 'loss_function': loss_option}
    if model_option == 'MLPLinear':
        config['lr'] = 3e-5
    if dataset == 'MNIST':
        config['input_size'] = 28 * 28
        if 'Prob' in config['model']:
            config['epoch_num'] = 50
        if config['labels']['pos'][0] == '3' and config['labels']['pos'][1] == '5':
            config['simple_train_batch_size'] = 500
        if config['model'] == 'CNN_MNIST' or config['model'] == 'ResNet':
            config['simple_train_batch_size'] = 500
    else:
        config['input_size'] = 3 * 32 * 32
        config['simple_train_batch_size'] = 500
    print('setting: ', config['labels']['all'])
    print('save data')
    save_data_binary(config, dir_path)
    for seed in range(1):
        experiments(seed, config, dir_path)


if __name__ == '__main__':
    run_experiments()
