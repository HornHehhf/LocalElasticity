import torch
import torch.nn as nn
import sys
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

from utils import set_random_seed, shuffle_in_union, get_minibatches_idx
from data import save_data_binary, load_data_binary_from_pickle
from models import autoencoder
from clustering import clustering_images_autoencoder


torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def build_autoencoder(config):
    model = autoencoder()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    return model, loss_function, optimizer


def train_batch_autoencoder(trainloader, model, loss_function, optimizer, config):
    model.train()
    for epoch in range(config['epoch_num']):
        if epoch % (config['epoch_num'] // 10) == 0:
            print('current epoch: ', epoch)
        total_loss = 0
        minibatches_idx = get_minibatches_idx(len(trainloader), minibatch_size=config['simple_train_batch_size'],
                                              shuffle=True)
        for minibatch in minibatches_idx:
            inputs = torch.Tensor(np.array([list(trainloader[x][0].cpu().numpy()) for x in minibatch]))
            targets = torch.Tensor(np.array([list(trainloader[x][1].cpu().numpy()) for x in minibatch]))
            inputs, targets = Variable(inputs.cuda()).squeeze(1), Variable(targets.float().cuda()).squeeze(1)
            # print(inputs.size())
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.size())
            loss = loss_function(outputs, inputs)
            total_loss += loss
            loss.backward()
            optimizer.step()
        if epoch % (config['epoch_num'] // 10) == 0:
            pic = to_img(outputs.cpu().data)
            save_image(pic, '/path/to/experiments/dir/figures/autoencoders/image_{}.png'.format(epoch))
            print('loss', total_loss)


def run_autoencoder_clustering():
    dataset = sys.argv[1].split('=')[1]
    pos_one = sys.argv[2].split('=')[1]
    pos_two = sys.argv[3].split('=')[1]
    neg_one = sys.argv[4].split('=')[1]
    neg_two = sys.argv[5].split('=')[1]
    cifar10_classes = {'plane': '0', 'car': '1', 'bird': '2', 'cat': '3', 'deer': '4', 'dog': '5', 'frog': '6',
                       'horse': '7', 'ship': '8', 'truck': '9'}
    dir_path = '/path/to/experiments/dir'
    config = {'batch_size': 1, 'epoch_num': 5000, 'lr': 0.001, 'test_batch_size': 1, 'sample_size': 1000,
              'simple_train_batch_size': 2000, 'simple_test_batch_size': 500,
              'labels': {'pos': [pos_one, pos_two], 'neg': [neg_one, neg_two],
                         'all': [pos_one, pos_two, neg_one, neg_two]}, 'repeat_num': 1, 'dataset': dataset}
    model_path = dir_path + '/models/' + config['dataset'] + '_' + "_".join(config['labels']['all']) + '_autoencoder.pt'
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
    print('build model')
    model, loss_function, optimizer = build_autoencoder(config)
    print('train model')
    train_batch_autoencoder(train_data, model, loss_function, optimizer, config)
    print('save model')
    torch.save(model.state_dict(), model_path)
    print('load model')
    model.load_state_dict(torch.load(model_path))
    print('clustering images')
    clustering_images_autoencoder(train_data, index_list, train_data_label[index_list], model)


if __name__ == '__main__':
    run_autoencoder_clustering()
