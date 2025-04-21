import ast
import configparser
import os
import time

width = os.get_terminal_size().columns

from SPP.spp import SPP

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('spp_params.ini.example')
    use_case = config['DEFAULT']['use_case']
    n_classes = int(config[use_case]['n_classes'])
    n_features = int(config[use_case]['n_features'])
    if 'fix_level' in config[use_case]:
        fix_level = int(config[use_case]['fix_level'])
    else:
        fix_level = None
    weights_file = config[use_case]['weights_file']
    f1_file = config[use_case]['f1_file']
    unwanted_classes = ast.literal_eval(config[use_case]['unwanted_classes'])
    test_classes_list = ast.literal_eval(config['TEST']['test_classes_list'])
    test_classes_list.sort()

    spp = SPP(n_classes=n_classes, n_features=n_features, unwanted_classes=unwanted_classes, use_case=use_case,
              weights_file=weights_file, fix_level=fix_level, f1_file=f1_file)

    ## Solve with Heuristic and compute time
    elapsed_times = []
    start = time.time()
    spp.solve_spp_greedy(save=False, show_plot_gain=False, print_console=True)
    end = time.time()
    elapsed_times.append(end - start)
    print(f'Elapsed time: {sum(elapsed_times) / len(elapsed_times)} s')