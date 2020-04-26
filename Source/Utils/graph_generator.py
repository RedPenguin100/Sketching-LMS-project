import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt


def _get_empty_result_dict():
    result_dict = {'ridge': {},
                   'ridge_boost': {},
                   'lasso': {},
                   'lasso_boost': {},
                   'elastic': {},
                   'elastic_boost': {},
                   }
    return result_dict


def parse_first_performance(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    result_dict = _get_empty_result_dict()
    for line in data:
        split_line = line.split(sep=',')
        alg, n, d, duration = split_line
        n_values, duration_values = result_dict[alg].get(int(d), ([], []))
        n_values.append(int(n))
        duration_values.append(float(duration))
        result_dict[alg][int(d)] = (n_values, duration_values)
    return result_dict


def parse_second_performance(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    result_dict = _get_empty_result_dict()
    for line in data:
        split_line = line.split(sep=',')
        alg, n, alphas_amount, duration = split_line
        n_values, duration_values = result_dict[alg].get(int(alphas_amount), ([], []))
        n_values.append(int(n))
        duration_values.append(float(duration))
        result_dict[alg][int(alphas_amount)] = (n_values, duration_values)
    return result_dict


def parse_third_performance(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    result_dict = _get_empty_result_dict()
    for line in data:
        split_line = line.split(sep=',')
        alg, dataset_name, alphas_amount, duration = split_line
        alphas_amounts, duration_values = result_dict[alg].get(dataset_name, ([], []))
        alphas_amounts.append(alphas_amount)
        duration_values.append(float(duration))
        result_dict[alg][dataset_name] = (alphas_amounts, duration_values)
    return result_dict


def plot_two_algorithms(result_dict, alg_name, x_name, y_name,
                        hyper_param_name):
    non_boost_dict = result_dict[alg_name]
    boost_dict = result_dict[alg_name + '_boost']
    fig, ax = plt.subplots()

    for hyper_param in boost_dict:
        x, y = boost_dict[hyper_param]
        ax.plot(x, y, '--', label='{}-boost, {h_param_name}={h_param}'.format(
            alg_name, h_param_name=hyper_param_name, h_param=hyper_param))
    for hyper_param in non_boost_dict:
        x, y = non_boost_dict[hyper_param]
        ax.plot(x, y, '^', label='{}, {h_param_name}={h_param}'.format(
            alg_name, h_param_name=hyper_param_name, h_param=hyper_param))

    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large',
                       prop={'size': 11})
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def plot_all_algs_third_performance(result_dict, dataset):
    fig, ax = plt.subplots()

    for algorithm, results in result_dict.items():
        x, y = results[dataset]
        if 'boost' in algorithm:
            ax.plot(x, y, '--', label=algorithm)
        else:
            ax.plot(x, y, '^', label=algorithm)
    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large',
                       prop={'size': 11})

    plt.xlabel('Number of alphas |$\\mathbb{A}$|')
    plt.ylabel('Computation time (seconds)')
    plt.show()


def parse_linreg_performance(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    result_dict = {'linreg': ([], [], []),
                   'linreg_boost': ([], [], [])}
    for line in data:
        split_line = line.split(sep=',')
        alg, n, d, duration = split_line
        n_s, d_s, durations = result_dict[alg]
        n_s.append(int(n))
        d_s.append(int(d))
        durations.append(float(duration))
        result_dict[alg] = (n_s, d_s, durations)
    return result_dict


def plot_linreg_performance(result_dict):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for algorithm, results in result_dict.items():
        if 'boost' in algorithm:
            (n_s, d_s, durations) = results
            ax.scatter(n_s, d_s, durations, marker='o', label=algorithm)
        else:
            (n_s, d_s, durations) = results
            ax.scatter(n_s, d_s, durations, marker='^', label=algorithm)
        # else:
        #     ax.plot(x, y, '^', label=algorithm)
    legend = ax.legend(loc='upper left', shadow=True, fontsize='x-large',
                       prop={'size': 11})

    plt.xlabel('Amount of Data n')
    plt.ylabel('Dimension d')
    ax.set_zlabel('Duration(seconds)')
    plt.show()


result_dict = parse_linreg_performance('..\\..\\Tests\\LINREG_PERFORMANCE_TEST_D_VALUES_very_large.txt')
plot_linreg_performance(result_dict)
result_dict = parse_linreg_performance('..\\..\\Tests\\LINREG_PERFORMANCE_TEST_D_VALUES_night_run.txt')
plot_linreg_performance(result_dict)
result_dict = parse_linreg_performance('..\\..\\Tests\\LINREG_PERFORMANCE_TEST_D_VALUES_large_values.txt')
plot_linreg_performance(result_dict)
