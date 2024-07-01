import logging
import os

from model_performance.performanceAnalyzer import calculate_F1_score
from model_analysis.modelAnalyzer import ModelAnalyzer
from setup_logger import logger
import multiprocessing as mp
from ast import literal_eval
from itertools import product
import pandas as pd

max_usable_cores = 24
# inference_points_list = list(range(2, 5))
inference_points_list = [3]

classes_filter = ['Amazon Echo', 'Android Phone', 'Belkin Wemo switch', 'Belkin wemo motion sensor', 'Dropcam',
                  'HP Printer', 'Insteon Camera', 'Laptop', 'Light Bulbs LiFX Smart Bulb', 'MacBook',
                  'NEST Protect smoke alarm', 'Netatmo Welcome', 'Netatmo weather station', 'PIX-STAR Photo-frame',
                  'Samsung Galaxy Tab', 'Samsung SmartCam', 'Smart Things', 'TP-Link Day Night Cloud camera',
                  'TP-Link Smart plug', 'Triby Speaker']

train_data_dir_path = '/home/ddeandres/UNSW_PCAPS/train/train_data_hybrid'
test_data_dir_path = '/home/ddeandres/UNSW_PCAPS/test/csv_files'
flow_counts_file_path = '/home/ddeandres/UNSW_PCAPS/hyb_code/16-10-05-flow-counts.csv'
results_dir_path = '/home/ddeandres/distributed_in_band/UNSW/cluster_model_analysis_results/correlation_analysis'

force_rewrite = False


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


def run_analysis(input_data):
    n_point = input_data[0]
    cluster_id = input_data[1]
    cluster_data_file_path = input_data[2]
    cluster_info = input_data[3]
    return __run_analysis(n_point, cluster_id, cluster_data_file_path, cluster_info)


def __run_analysis(n_point, cluster_id, cluster_data_file_path, cluster_info):
    base = os.path.basename(cluster_data_file_path)
    stem = os.path.splitext(base)[0]
    exp_id = stem.split('_')[1]
    logger = logging.getLogger(f'UNSW.{exp_id}.analyzer_{cluster_id}_{n_point}')
    logger.info(f"Starting analysis of: Cluster id: {cluster_id}, npoint {n_point}")
    f_name = f"{results_dir_path}/{stem}/unsw_models_{n_point}pkts_PF_WB_20CL_Cluster{cluster_id}.csv"
    model_analyzer = ModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_file_path,
                                   classes_filter, cluster_data_file_path, logger)
    model_analyzer.load_cluster_data(cluster_info.loc[cluster_id])
    model_analyzer.analyze_model_n_packets(n_point, f_name, force_rewrite)
    logger.info(f"Finished analyzing n={n_point}, Cluster={cluster_id}. Results at: {results_dir_path}")


def run_experiment(folder):
    directory = os.fsencode(folder)

    for file in os.listdir(directory):
        file_string = file.decode("utf-8")
        solution_file_path = os.path.join(folder, file_string)
        if os.path.isdir(solution_file_path):
            # skip directories
            continue
        base = os.path.basename(file_string)
        stem = os.path.splitext(base)[0]
        extension = os.path.splitext(base)[1]
        if 'csv' not in extension:
            continue
        exp_id = stem.split('_')[1]
        logger = logging.getLogger(f'UNSW.{exp_id}')
        logger.info(f"Starting experiment: {exp_id}")

        # Create folder for saving results
        results_folder = os.path.join(folder, stem)
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        cluster_info = pd.read_csv(solution_file_path,
                                   converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
        cluster_id_list = cluster_info['Cluster'].to_list()
        consumed_cores = min([max_usable_cores, len(inference_points_list) * len(cluster_id_list)])
        logging.getLogger(f'UNSW').info(f'Will use {consumed_cores} cores. Starting pool...')
        input_data = list(product(inference_points_list, cluster_id_list, [str(solution_file_path)], [cluster_info]))
        with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
            try:
                # issue tasks to the process pool
                pool.imap_unordered(run_analysis, input_data)
                # for result in pool.imap_unordered(run_analysis, input_data):
                #     pass  # or do something with result, if pool_tasks returns a value
                # shutdown the process pool
                pool.close()
            except KeyboardInterrupt:
                logger.error("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
            # wait for all issued task to complete
            pool.join(90)
            try:
                score = calculate_F1_score(f'{folder}/{file_string}', str(results_folder))
                logger.info(f"F1 score: {score}")
            except ValueError as e:
                logger.error(f"F1 score could not be calculated. The following error was raised: {e}")
        del pool
        logger.info(f"Finished experiment: {exp_id}")


def main():
    logger.info("Starting program.")
    run_experiment(results_dir_path)
    logger.info("Finished program.")


if __name__ == '__main__':
    raise SystemExit(main())
