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


# result_dict = parse_first_performance('..\\..\\Tests\\FIRST_PERFORMANCE_TEST_RESULTS.txt')
result_dict = parse_third_performance('..\\..\\Tests\\THIRD_PERFORMANCE_TEST_RESULTS.txt')
plot_all_algs_third_performance(result_dict, '3D_spatial_network')
plot_all_algs_third_performance(result_dict, 'household_power_consumption')
# plot_two_algorithms(result_dict, 'ridge', 'Data size n', 'Computation time (seconds)', '|alphas|')
# plot_two_algorithms(result_dict, 'lasso', 'Data size n', 'Computation time (seconds)', '|alphas|')
