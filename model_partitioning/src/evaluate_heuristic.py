# # Read cluster solution from CSV file.
import configparser
import os
import pandas as pd
from model_partitioning.src.SPP.SPP import SPP, literal_converter
import ast
directory_in_str = '/home/ddeandres/spp_features_and_classes/heuristic_results'
directory = os.fsencode(directory_in_str)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('spp_params.ini')
    use_case = config['DEFAULT']['use_case']
    n_classes = int(config[use_case]['n_classes'])
    n_features = int(config[use_case]['n_features'])
    weights_file = config[use_case]['weights_file']
    f1_file = config[use_case]['f1_file']
    unwanted_classes = ast.literal_eval(config['DEFAULT']['unwanted_classes'])

    heuristic_costs = []
    spp = SPP(n_classes=n_classes, n_features=n_features, weights_file=weights_file, f1_file=f1_file,
              unwanted_classes=unwanted_classes)
    for file in os.listdir(directory):
        cluster_info = pd.read_csv(f'{directory_in_str}/{file.decode("utf-8")}', converters=dict.fromkeys(['Class List','Feature List'], literal_converter))
        cluster_info = cluster_info.reset_index(drop=True).drop(columns=['Unnamed: 0']).set_index('Cluster')
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        heuristic_costs.append(sol_cost)
    print(heuristic_costs)