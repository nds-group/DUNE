from model_performance.performanceAnalyzer import calculate_F1_score
from model_analysis.modelAnalyzer import ModelAnalyzer
import multiprocessing as mp
from ast import literal_eval
from itertools import product
import pandas as pd
import logging

logger = logging.getLogger('UNSW')

classes_filter = ['Amazon Echo', 'Android Phone', 'Belkin Wemo switch', 'Belkin wemo motion sensor', 'Dropcam',
                  'HP Printer', 'Insteon Camera', 'Laptop', 'Light Bulbs LiFX Smart Bulb', 'MacBook',
                  'NEST Protect smoke alarm', 'Netatmo Welcome', 'Netatmo weather station', 'PIX-STAR Photo-frame',
                  'Samsung Galaxy Tab', 'Samsung SmartCam', 'Smart Things', 'TP-Link Day Night Cloud camera',
                  'TP-Link Smart plug', 'Triby Speaker']

train_data_dir_path = '/home/ddeandres/UNSW_PCAPS/train/train_data_hybrid'
test_data_dir_path = '/home/ddeandres/UNSW_PCAPS/test/csv_files'
flow_counts_file_path = '/home/ddeandres/UNSW_PCAPS/hyb_code/16-10-05-flow-counts.csv'
# cluster_data_file_path = '/home/ddeandres/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv'
# results_dir_path = '/home/ddeandres/distributed_in_band/UNSW/cluster_model_analysis_results/test'

experiment_nr = '20CL_6_SPP_solution'
cluster_data_file_path = f'/home/ddeandres/distributed_in_band/UNSW/cluster_model_analysis_results/correlation_analysis/{experiment_nr}.csv'
results_dir_path = f'/home/ddeandres/distributed_in_band/UNSW/cluster_model_analysis_results/correlation_analysis/{experiment_nr}'


force_rewrite = False


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


cluster_info = pd.read_csv(cluster_data_file_path,
                           converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))


def run_analysis(input_data):
    n_point = input_data[0]
    cluster_id = input_data[1]
    return __run_analysis(n_point, cluster_id)

def __run_analysis(n_point, cluster_id):
    print(f"Starting analysis of: Cluster id: {cluster_id}, npoint {n_point}")
    f_name = f"{results_dir_path}/unsw_models_{n_point}pkts_PF_WB_20CL_Cluster{cluster_id}.csv"
    model_analyzer = ModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_file_path,
                                   classes_filter, cluster_data_file_path, logger)
    model_analyzer.load_cluster_data(cluster_info.loc[cluster_id])
    model_analyzer.analyze_model_n_packets(n_point, f_name, force_rewrite)
    print(f"Finished analyzing n={n_point}, Cluster={cluster_id}. Results at: {results_dir_path}")


def main():
    # inference_points_list = list(range(2, 5))
    inference_points_list = [3]
    cluster_id_list = cluster_info['Cluster'].to_list()
    # cluster_id_list = [0, 3, 6]
    consumed_cores = min([24, len(inference_points_list) * len(cluster_id_list)])
    print(f'Will use {consumed_cores} cores. Starting pool...')

    with mp.get_context('spawn').Pool(processes=consumed_cores) as pool:
        input_data = list(product(inference_points_list, cluster_id_list))
        for result in pool.imap_unordered(run_analysis, input_data):
            pass  # or do something with result, if pool_tasks returns a value
        # pool.starmap(__run_analysis, list(product(inference_points_list, cluster_id_list)), chunksize=1)

    with mp.get_context('spawn').Pool(processes=consumed_cores) as pool:
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
        pool.join()
        try:
            score = calculate_F1_score(cluster_data_file_path, results_dir_path)
            print(f"F1 score: {score}")
        except ValueError as e:
            logger.error(f"F1 score could not be calculated. The following error was raised: {e}")
    del pool

if __name__ == '__main__':
    raise SystemExit(main())
