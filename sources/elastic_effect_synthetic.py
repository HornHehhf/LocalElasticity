import torch
import numpy as np
import sys
from scipy.stats import pearsonr

from utils import set_random_seed, get_data, plot_scatter, plot_delta_versus_distance
from data import two_cosine, double_helix, donut, two_sphere, two_fold_surface, two_cycles
from elastic_effect import build_model, simple_train_batch, simple_test_batch, train_elastic_effect, \
    get_similarity_matrix

torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def clip_value(x, min_x=-1, max_x=1):
    if x < min_x:
        print('x', x)
        return min_x
    elif x > max_x:
        print('x', x)
        return max_x
    else:
        return x


def get_distance(original_positions, index_list, option='donut'):
    total_distance = {}
    for idx in range(len(original_positions)):
        if idx not in index_list:
            continue
        if 'two_sphere' in option:
            cur_distance = []
            for j in range(len(index_list)):
                cur_distance.append(np.arccos(clip_value(np.sum(original_positions[index_list[j]] * original_positions[idx]))))
            total_distance[idx] = cur_distance
        elif option == 'donut' or option == 'two_cycles':
            cur_distance = []
            for j in range(len(index_list)):
                direct_distance = np.abs(original_positions[index_list[j]] - original_positions[idx])
                cur_distance.append(np.min([direct_distance, 2 * np.pi - direct_distance]))
            total_distance[idx] = cur_distance
        elif option == 'two_fold_surface':
            cur_distance = []
            for j in range(len(index_list)):
                x_1, y_1, z_1, max_x_1 = original_positions[idx]
                x_2, y_2, z_2, max_x_2 = original_positions[index_list[j]]
                if y_1 * y_2 > 0:
                    direct_distance = np.sqrt(y_1 * y_1 + 1) * np.abs(x_1 - x_2)
                else:
                    direct_distance = np.sqrt(y_1 * y_1 + 1) * (2 * max_x_1 - x_1 - x_2)
                    total_length = np.sqrt(y_1 * y_1 + 1) * 4 * max_x_1
                    direct_distance = np.min([direct_distance, total_length - direct_distance])
                distance = np.sqrt(direct_distance * direct_distance + (z_1 - z_2) * (z_1 - z_2))
                cur_distance.append(distance)
            total_distance[idx] = cur_distance
        elif len(original_positions.shape) == 1:
            total_distance[idx] = np.abs(original_positions[index_list] - original_positions[idx])
        else:
            total_distance[idx] = np.linalg.norm(original_positions[index_list] - original_positions[idx], axis=1)
    return total_distance


def get_delta_distance(total_delta, total_distance, index_list):
    delta = []
    distance = []
    for idx in index_list:
        delta.append(list(total_delta[idx]))
        distance.append(list(total_distance[idx]))
    return delta, distance


def get_kernel_distance(similarity_matrix, index_list, dim=10, gate=0.1):
    eigen_values, eigen_vectors = np.linalg.eig(similarity_matrix)
    sorted_indices = np.argsort(eigen_values)
    sorted_indices = sorted_indices[::-1]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    eigen_values = eigen_values[sorted_indices]
    print('positive eigenvalues', np.sum(eigen_values >= 0.0))
    print('eigenvalues >= 0.1', np.sum(eigen_values >= gate))
    print('eigenvalues >= -0.1', np.sum(eigen_values >= -gate))
    # dim = np.sum(eigen_values >= 0.01)
    topk_evecs = eigen_vectors[:, :dim]
    topk_evals = eigen_values[:dim]
    print('the smallest chosen eigenvalue', topk_evals[-1])
    diagonal_matrix = np.zeros((dim, dim))
    for index in range(dim):
        diagonal_matrix[index][index] = np.sqrt(topk_evals[index])
    pca_features = np.dot(topk_evecs, diagonal_matrix)
    print('pca_features size', pca_features.shape)

    kernel_distance = {}
    for idx in range(len(similarity_matrix)):
        cur_distance = np.zeros(len(similarity_matrix))
        for idy in range(len(similarity_matrix)):
            cur_distance[idy] = np.linalg.norm(pca_features[idx]-pca_features[idy])
        kernel_distance[index_list[idx]] = cur_distance
    return kernel_distance


def run_elastic_effect_synthetic():
    data_option = sys.argv[1].split('=')[1]
    method_option = sys.argv[2].split('=')[1]
    model_option = sys.argv[3].split('=')[1]
    loss_option = sys.argv[4].split('=')[1]
    theta_option = sys.argv[5].split('=')[1]
    set_random_seed(666)
    dir_path = '/path/to/experiments/dir'
    model_path = dir_path + 'models/' + data_option + '_' + model_option + '.pt'
    figure_path = dir_path + 'figures/' + data_option + '.png'
    result_figure_path = dir_path + 'figures/' + data_option + '_' + model_option + '_' + method_option + '_' + \
                         theta_option + '.png'
    config = {'batch_size': 1, 'epoch_num': 20000, 'lr': 1e-5, 'test_batch_size': 1, 'sample_size': 100,
              'simple_train_batch_size': 100, 'simple_test_batch_size': 10, 'prob_lr': 1e-6, 'input_size': 3,
              'loss_function': loss_option, 'method_option': method_option, 'model': model_option, 'optimizer': 'SGD'}
    if data_option == 'two_cosine':
        config['input_size'] = 2
        # config['lr'] = 1e-4
        positions, labels, original_positions = two_cosine(config['sample_size'])
        positions_large, labels_large, original_positions_large = two_cosine(10000)
        dataloader = get_data(positions, labels)
        dataloader_large = get_data(positions_large, labels_large)
        print('plot data functions')
        plot_scatter(positions_large, labels_large, figure_path)
    elif data_option == 'two_cycles':
        config['input_size'] = 2
        # config['lr'] = 1e-4
        positions, labels, original_positions = two_cycles(config['sample_size'])
        positions_large, labels_large, original_positions_large = two_cycles(10000)
        dataloader = get_data(positions, labels)
        dataloader_large = get_data(positions_large, labels_large)
        print('plot data functions')
        plot_scatter(positions_large, labels_large, figure_path)
    elif data_option == 'two_fold_surface':
        config['input_size'] = 3
        # config['lr'] = 1e-6
        positions, labels, original_positions = two_fold_surface(config['sample_size'])
        positions_large, labels_large, original_positions_large = two_fold_surface(10000)
        dataloader = get_data(positions, labels)
        dataloader_large = get_data(positions_large, labels_large)
        print('plot data functions')
        plot_scatter(positions_large, labels_large, figure_path, '3D')
    elif data_option == 'double_helix':
        config['input_size'] = 3
        positions, labels, original_positions = double_helix(config['sample_size'])
        positions_large, labels_large, original_positions_large = double_helix(10000)
        dataloader = get_data(positions, labels)
        dataloader_large = get_data(positions_large, labels_large)
        print('plot data functions')
        plot_scatter(positions_large, labels_large, figure_path, '3D')
    elif data_option == 'two_sphere':
        config['input_size'] = 3
        if config['model'] == 'MLPProb':
            config['lr'] = 3e-3
        elif config['model'] == 'MLPLinearProb':
            config['lr'] = 3e-4
        positions, labels, original_positions = two_sphere(config['sample_size'])
        positions_large, labels_large, original_positions_large = two_sphere(10000)
        dataloader = get_data(positions, labels)
        dataloader_large = get_data(positions_large, labels_large)
        print('plot data functions')
        plot_scatter(positions_large, labels_large, figure_path, '3D')
    else:
        config['input_size'] = 3
        positions, labels, original_positions = donut(config['sample_size'])
        positions_large, labels_large, original_positions_large = donut(10000)
        dataloader = get_data(positions, labels)
        dataloader_large = get_data(positions_large, labels_large)
        print('plot data functions')
        plot_scatter(positions_large, labels_large, figure_path, '3D')
    print(config['lr'])
    print('build model')
    model, loss_function, optimizer = build_model(config)
    if theta_option == 'optimal':
        print('train model')
        simple_train_batch(dataloader, model, loss_function, optimizer, config)
        print('save model')
        torch.save(model.state_dict(), model_path)
        print('load model')
        model.load_state_dict(torch.load(model_path))
    data_res, data_np = simple_test_batch(dataloader, model, config)
    print('data accuracy', data_res)
    index_list = []
    for index in range(len(dataloader)):
        if dataloader[index][1] == 1:
            index_list.append(index)
    config['epoch_num'] = 1
    total_delta = train_elastic_effect(dataloader, model, loss_function, config, index_list)
    if config['method_option'] == 'kernel':
        similarity_matrix = get_similarity_matrix(total_delta, index_list)
        total_delta = get_kernel_distance(similarity_matrix, index_list)
    data_res, data_np = simple_test_batch(dataloader, model, config)
    print('data accuracy', data_res)
    total_distance = get_distance(original_positions, index_list, option=data_option)
    delta, distance = get_delta_distance(total_delta, total_distance, index_list)
    print('average delta', np.mean(np.array(delta)))
    print('plot result figures')
    plot_delta_versus_distance(delta, distance, result_figure_path)
    print('delta', delta)
    print('distance', distance)
    print('Pearson Correlation')
    print(pearsonr(np.array(delta).reshape(-1), np.array(distance).reshape(-1))[0])


if __name__ == '__main__':
    run_elastic_effect_synthetic()
