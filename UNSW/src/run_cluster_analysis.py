from model_performance.performanceAnalyzer import calculate_F1_score
from model_analysis.modelAnalyzer import ModelAnalyzer
from multiprocessing import Pool
from ast import literal_eval
from itertools import product
import pandas as pd

classes_filter = ['Amazon Echo', 'Android Phone', 'Belkin Wemo switch', 'Belkin wemo motion sensor', 'Dropcam',
                  'HP Printer', 'Insteon Camera', 'Laptop', 'Light Bulbs LiFX Smart Bulb', 'MacBook',
                  'NEST Protect smoke alarm', 'Netatmo Welcome', 'Netatmo weather station', 'PIX-STAR Photo-frame',
                  'Samsung Galaxy Tab', 'Samsung SmartCam', 'Smart Things', 'TP-Link Day Night Cloud camera',
                  'TP-Link Smart plug', 'Triby Speaker']

train_data_dir_path = '/home/nds-admin/UNSW_PCAPS/train/train_data_hybrid'
test_data_dir_path = '/home/nds-admin/UNSW_PCAPS/test/csv_files'
flow_counts_file_path = '/home/nds-admin/UNSW_PCAPS/hyb_code/16-10-05-flow-counts.csv'
cluster_data_file_path = '/home/ddeandres/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv'
results_dir_path = '/home/ddeandres/distributed_in_band/UNSW/cluster_model_analysis_results/test'

force_rewrite = False


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)


cluster_info = pd.read_csv(cluster_data_file_path,
                           converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))


def run_analysis(input_data):
    n_point = input_data[0]
    cluster_id = input_data[1]
    print(f"Starting analysis of: Cluster id: {cluster_id}, npoint {n_point}")
    f_name = f"{results_dir_path}/unsw_models_{n_point}pkts_PF_WB_20CL_Cluster{cluster_id}.csv"
    model_analyzer = ModelAnalyzer(train_data_dir_path, test_data_dir_path, flow_counts_file_path,
                                   classes_filter, cluster_data_file_path)
    model_analyzer.load_cluster_data(cluster_info.loc[cluster_id])
    model_analyzer.analyze_model_n_packets(n_point, f_name, force_rewrite)
    print(f"Finished analyzing n={n_point}, Cluster={cluster_id}. Results at: {results_dir_path}")


inference_points_list = list(range(2, 5))
cluster_id_list = cluster_info['Cluster'].to_list()
consumed_cores = min([32, len(inference_points_list)*len(cluster_id_list)])
print(f'Will use {consumed_cores} cores. Starting pool...')

with Pool(processes=consumed_cores) as pool:
    for result in pool.imap_unordered(run_analysis, list(product(inference_points_list, cluster_id_list))):
        pass  # or do something with result, if pool_tasks returns a value
#     # pool.starmap(run_analysis, zip(list(range(2, 5)), list(range(3))), chunksize=1)


print(calculate_F1_score(cluster_data_file_path, results_dir_path))
