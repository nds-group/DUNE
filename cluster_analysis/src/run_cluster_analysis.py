import ast
import configparser
import os
from os import path

from cluster_analysis.src.model_performance.performanceAnalyzer import calculate_TOTAL_TCAM_usage
from model_analysis.modelAnalyzer import UNSWModelAnalyzer, TONModelAnalyzer
from model_performance.performanceAnalyzer import calculate_f1_score, select_best_models_per_cluster, \
    append_best_models_info_to_cluster_info, generate_score_per_class_report_for_best_models
import multiprocessing as mp
from ast import literal_eval
from itertools import product, chain
import pandas as pd
import logging

cluster_info = None

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


def run_analysis(input_data):
    n_point = input_data[0]
    cluster_id = input_data[1]
    return __run_analysis(n_point, cluster_id)

def __run_analysis(n_point, cluster_id):
    print(f"Starting analysis of: Cluster id: {cluster_id}, npoint {n_point}")
    f_name = f"{results_dir_path}/unsw_models_{n_point}pkts_PF_WB_20CL_Cluster{cluster_id}.csv"
    model_analyzer = None
    if use_case == 'UNSW':
        model_analyzer = UNSWModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_file_path,
                                           classes_filter, cluster_data_file_path, logger)
    elif use_case == 'TON-IOT':
        model_analyzer = TONModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_train_file_path,
                                          flow_counts_test_file_path, classes_filter, cluster_data_file_path, logger)
    model_analyzer.load_cluster_data(cluster_info.loc[cluster_id])
    model_analyzer.analyze_model_n_packets(n_point, f_name, force_rewrite, grid_search)
    print(f"Finished analyzing n={n_point}, Cluster={cluster_id}. Results at: {results_dir_path}")


def main():
    global cluster_info
    cluster_id_list = cluster_info['Cluster'].to_list()
    consumed_cores = min([max_usable_cores, len(inference_points_list) * len(cluster_id_list)])
    print(f'Will use {consumed_cores} cores. Starting pool...')

    with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
        input_data = list(product(inference_points_list, cluster_id_list))
        try:
            # issue tasks to the process pool
            pool.imap_unordered(run_analysis, input_data, chunksize=chunksize)
            # shutdown the process pool
            pool.close()
        except KeyboardInterrupt:
            logger.error("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        # wait for all issued task to complete
        pool.join()
        try:
            # cluster_info = pd.read_csv(cluster_data_file_path, converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
            cluster_info = cluster_info.drop(['Unnamed: 0'], axis=1)
            cluster_info = cluster_info.set_index('Cluster', drop=True)
            classes = cluster_info['Class List'].sum()
            classes.sort()

            if use_case == 'TON-IOT':
                flow_pkt_counts = pd.read_csv(flow_counts_test_file_path)
                support = flow_pkt_counts['type'].value_counts().loc[classes].sort_index()
            else:
                flow_pkt_counts = pd.read_csv(flow_counts_file_path)
                support = flow_pkt_counts['label'].value_counts().loc[classes].sort_index()

            best_models_df = select_best_models_per_cluster(cluster_info, results_dir_path, support)
            cluster_info = append_best_models_info_to_cluster_info(cluster_info, best_models_df)
            score_per_class_df = generate_score_per_class_report_for_best_models(classes, best_models_df, support)

            # Create folder for saving results
            if not os.path.exists(f'{results_dir_path}/perf_results'):
                os.makedirs(f'{results_dir_path}/perf_results')

            cluster_info.to_csv(f'{results_dir_path}/perf_results/cluster_info_df.csv')
            score_per_class_df.to_csv(f'{results_dir_path}/perf_results//score_per_cluster_per_class_df.csv')

            score = calculate_f1_score(score_per_class_df)
            print(f"F1 score: {score}")
            tcam = calculate_TOTAL_TCAM_usage(cluster_info)
            print(f"TOTAL TCAM usage: {tcam}")
        except ValueError as e:
            logger.error(f"F1 score could not be calculated. The following error was raised: {e}")
    del pool


if __name__ == '__main__':
    basepath = path.dirname(__file__)
    logging.basicConfig()
    config = configparser.ConfigParser()
    config.read(path.abspath(path.join(basepath,'params.ini')))
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)
    force_rewrite = config['DEFAULT']['force_rewrite'] == 'True'
    grid_search = bool(config['DEFAULT']['grid_search'])
    max_usable_cores = int(config['DEFAULT']['max_usable_cores'])
    chunksize = int(config['DEFAULT']['chunksize'])
    train_data_dir_path = config[use_case]['train_data_dir_path']
    test_data_dir_path = config[use_case]['test_data_dir_path']
    inference_points_list = ast.literal_eval(config[use_case]['inference_point_list'])
    if use_case == 'UNSW':
        flow_counts_file_path = config[use_case]['flow_counts_file_path']
    elif use_case == 'TON-IOT':
        flow_counts_test_file_path = config[use_case]['flow_counts_test_file_path']
        flow_counts_train_file_path = config[use_case]['flow_counts_train_file_path']
    else:
        raise ValueError("use_case can only be 'UNSW' or 'TON-IOT'")
    cluster_data_file_path = config[use_case]['cluster_data_file_path']
    results_dir_path = config[use_case]['results_dir_path']
    classes_filter = ast.literal_eval(config[use_case]['classes_filter'])
    cluster_info = pd.read_csv(cluster_data_file_path,
                               converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
    raise SystemExit(main())
