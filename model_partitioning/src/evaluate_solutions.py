import configparser
import os
import pandas as pd
from SPP.SPP import SPP, literal_converter
import ast
directory_in_str = input("Enter the directory path where the solutions are stored: ")
directory = os.fsencode(directory_in_str)

"""
This script reads SPP solutions provided in CSV files and evaluates them with the objective function from the SPP class.
"""

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('spp_params.ini')
    use_case = config['DEFAULT']['use_case']
    n_classes = int(config[use_case]['n_classes'])
    n_features = int(config[use_case]['n_features'])
    weights_file = config[use_case]['weights_file']
    f1_file = config[use_case]['f1_file']
    unwanted_classes = ast.literal_eval(config[use_case]['unwanted_classes'])

    heuristic_costs = []
    spp = SPP(n_classes=n_classes, n_features=n_features, unwanted_classes=unwanted_classes, use_case=use_case,
              weights_file=weights_file, f1_file=f1_file)
    for file in os.listdir(directory):
        cluster_info = pd.read_csv(f'{directory_in_str}/{file.decode("utf-8")}',
                                   converters=dict.fromkeys(['Class List','Feature List'], literal_converter))
        cluster_info = cluster_info.set_index('Cluster')
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        heuristic_costs.append(sol_cost)
    print(heuristic_costs)