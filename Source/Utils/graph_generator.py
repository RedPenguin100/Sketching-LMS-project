import matplotlib.pyplot as plt


def parse_first_performance(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()

    result_dict = {'ridge': {},
                   'ridge_boost': {},
                   'lasso': {},
                   'lasso_boost': {},
                   'elastic': {},
                   'elastic_boost': {},
                   }

    for line in data:
        split_line = line.split(sep=',')
        alg, n, d, duration = split_line
        result_dict[alg][int(d)] = (int(n), float(duration))

    return result_dict
