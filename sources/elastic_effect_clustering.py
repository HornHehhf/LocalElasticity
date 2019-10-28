import sys
import numpy as np
import torch

from utils import set_random_seed, shuffle_in_union
from data import save_data_binary, load_data_binary_from_pickle
from elastic_effect import build_model, simple_test_batch, train_elastic_effect, get_similarity_matrix
from clustering import clustering_images, clustering_pca_images, clustering_images_resnet152, \
    similarity_clustering, kernel_clustering

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_tensor_type('torch.cuda.FloatTensor')


def elastic_effect_clustring(seed, data, index_list, config, dir_path):
    print('seed', seed)
    set_random_seed(seed)
    if 'Prob' in config['model'] and config['dataset'] == 'MNIST':
        model_path = dir_path + '/models/' + config['dataset'] + '_' + "_".join(config['labels']['all']) + '_' + \
                     str(seed) + '_' + config['model'] + '_50epoch.pt'
    else:
        model_path = dir_path + '/models/' + config['dataset'] + '_' + "_".join(config['labels']['all']) + '_' + \
                     str(seed) + '_' + config['model'] + '.pt'
    print('build model')
    model, loss_function, optimizer = build_model(config)
    if config['theta_option'] == 'optimal':
        print('load model')
        model.load_state_dict(torch.load(model_path))
    data_res, data_np = simple_test_batch(data, model, config)
    print('data accuracy', data_res)
    config['epoch_num'] = 1
    total_delta = train_elastic_effect(data, model, loss_function, config, index_list)
    similarity_matrix = get_similarity_matrix(total_delta, index_list)
    data_res, data_np = simple_test_batch(data, model, config)
    print('data accuracy', data_res)
    return similarity_matrix


def run_elastic_effect_clustering():
    dataset = sys.argv[1].split('=')[1]
    method_option = sys.argv[2].split('=')[1]
    model_option = sys.argv[3].split('=')[1]
    loss_option = sys.argv[4].split('=')[1]
    pos_one = sys.argv[5].split('=')[1]
    pos_two = sys.argv[6].split('=')[1]
    neg_one = sys.argv[7].split('=')[1]
    neg_two = sys.argv[8].split('=')[1]
    theta_option = sys.argv[9].split('=')[1]
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    dir_path = '/path/to/experiments/dir'
    config = {'batch_size': 1, 'epoch_num': 500, 'lr': 0.0003, 'test_batch_size': 1, 'sample_size': 1000,
              'simple_train_batch_size': 50, 'simple_test_batch_size': 50,
              'labels': {'pos': [pos_one, pos_two], 'neg': [neg_one, neg_two],
                         'all': [pos_one, pos_two, neg_one, neg_two]},
              'theta_option': theta_option, 'method_option': method_option, 'prob_lr': 1e-6, 'repeat_num': 1,
              'dataset': dataset, 'model': model_option, 'loss_function': loss_option}
    if dataset == 'MNIST':
        config['input_size'] = 28 * 28
    else:
        config['input_size'] = 3 * 32 * 32
    print('setting: ', config['labels']['all'])
    set_random_seed(666)
    print('save data')
    save_data_binary(config, dir_path)
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
    clustering_pca_images(train_data, index_list, train_data_label[index_list])
    # clustering_images_resnet152(train_data, index_list, train_data_label[index_list])

    similarity_matrix = np.zeros((len(index_list), len(index_list)))
    for seed in range(config['repeat_num']):
        similarity_matrix += elastic_effect_clustring(seed, train_data, index_list, config, dir_path)
    similarity_matrix /= config['repeat_num']
    print(train_data_label[index_list])
    print(similarity_matrix)
    if config['method_option'] == 'ripple':
        print('similarity clustering')
        similarity_clustering(similarity_matrix, train_data_label[index_list])
    else:
        print('kernel clustering')
        kernel_clustering(similarity_matrix, train_data_label[index_list])


if __name__ == '__main__':
    run_elastic_effect_clustering()
