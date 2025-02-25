import ast
import configparser
import os
import time

width = os.get_terminal_size().columns


from SPP.SPP import SPP

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('spp_params.ini')
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

    ## Evaluate cost for a given cluster
    # s_j = np.random.randint(2, size=n_classes)
    # s_j = np.zeros(n_classes)
    # s_j[11 - 1] = 1
    # s_j[22 - 1] = 1
    # print(s_j)
    # print(compute_group_gain_c4(s_j, spp.W, spp.F))

    ## Solve using Analytical Model (not possible without lineal cost/gain function)
    # spp.compute_costs(compute_group_gain)
    # spp.solve_SPP_with_ILP(save=False, minimize=False)

    ## Evaluate a random solution
    # cluster_info = spp.generate_random_spp_solution(7)
    # sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
    # print("The cost of the random partition:", sol_cost, end='\n')

    ## Solve Brute Force
    # timestamps = []
    # gains = []
    # classes_range = range(2, 11 + 1)
    # for n_classes in classes_range:
    #     print(f'Solving {n_classes} classes...')
    #     spp = SPP(n_classes=n_classes, n_features=n_features, unwanted_classes=unwanted_classes, weights_file=weights_file,
    #               f1_file=f1_file, name=use_case, fix_level=fix_level)
    #     # spp.compute_costs(compute_group_gain_c4)
    #     start = time.time()
    #     partition, gain = spp.solve_SPP_brute_force()
    #     end = time.time()
    #     gains.append(gain)
    #     timestamps.append(end - start)
    # plt.plot(list(classes_range), timestamps)
    # plt.show()
    # plt.plot(list(classes_range), gains)
    # plt.show()
    # spp.solve_SPP_brute_force()