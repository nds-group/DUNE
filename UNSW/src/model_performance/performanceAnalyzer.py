import json
from ast import literal_eval
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import os
import re

flow_pkts_data_file_path = "/home/nds-admin/UNSW_PCAPS/hyb_code/16-10-05-flow-counts.csv"

# list of all extracted features
feats_all = ["ip.len", 'ip.hdr_len', "ip.ttl", "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.push", "tcp.flags.fin",
             "tcp.flags.rst", "tcp.flags.reset", "tcp.flags.ece", "ip.proto", "srcport", "dstport",
             "tcp.window_size_value", "tcp.hdr_len", "udp.length", 'UDP Len Min', 'UDP Len Max', 'UDP Len Total',
             "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Total",
             "Flow IAT Min", "Flow IAT Max", "Flow IAT Mean", "Flow Duration", "SYN Flag Count", "ACK Flag Count",
             "PSH Flag Count", "FIN Flag Count", "RST Flag Count", "ECE Flag Count", "Packet Count"]

feats_sizes = [16, 16, 8, 1, 1, 1, 1, 1, 1,1, 8, 16, 16, 16, 4, 16, 16, 16, 16,16, 16, 16, 16, 32, 32, 32, 32, 8, 8, 8,
               8, 8, 8, 16]

available_TCAM_table = 24 * 12

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)

def convert_str_to_dict(field_value):
    return json.loads(field_value.replace("\'", "\""))

def calculate_score(support_total, mult_score_support, score):
    macro_score = np.mean(np.array(score))
    weighted_score = np.sum(mult_score_support) / support_total

    return macro_score, weighted_score


def concat_csvs(path, dir_entries):
    d_frames = []
    count = 2

    for dir_csv in dir_entries:
        f_name = path + dir_csv
        i_df = pd.read_csv(f_name, sep=';')

        i_df = i_df.loc[~((i_df['tree'] == 5) & (i_df['no_feats'] > 4))]
        i_df = i_df.loc[~((i_df['tree'] == 3) & (i_df['no_feats'] > 7))]
        i_df = i_df[i_df['tree'] != 2]
        i_df = i_df[i_df['tree'] != 4]

        i_df = i_df[['depth', 'tree', 'no_feats', 'New_Macro_F1', 'New_Weighted_F1', 'N_Leaves', 'feats']]
        i_df = i_df[i_df['tree'] < 6]
        i_df = i_df[i_df['no_feats'] < 11]
        i_df = i_df[i_df['N_Leaves'] > 120]
        i_df['N'] = count
        i_df['F1_score'] = i_df.apply(lambda x: 0.5 * (x['New_Macro_F1'] + x['New_Weighted_F1']), axis=1)
        d_frames.append(i_df)
        count = count + 1

    return pd.concat(d_frames)


def calculate_tcam_for_codetables(models_df):
    models_df['tcam_codetable'] = models_df.apply(lambda x: (int(x['N_Leaves'] / 44) + 1) * x['tree'], axis=1)

    return models_df


def calculate_tcam_for_featable(models_df):
    feats_size_dict = {}
    for f_ind in range(0, len(feats_all)):
        feats_size_dict[feats_all[f_ind]] = feats_sizes[f_ind]
    feats_for_all_models = list(models_df['feats'])
    tcam_usage_per_model = []
    for feats_in_models in feats_for_all_models:
        tcam_usage = 0
        for feat in feats_in_models:
            bit_length = feats_size_dict[feat]
            tcam_usage = tcam_usage + int((2 * bit_length) / 44) + 1

        tcam_usage_per_model.append(tcam_usage)
    models_df['tcam_feature_table'] = tcam_usage_per_model
    return models_df


def total_tcam_usage(models_df):
    models_df['total_tcam_tbl_usage'] = models_df.apply(lambda x: x['tcam_feature_table'] + x['tcam_codetable'], axis=1)
    models_df['total_tcam_usage'] = models_df.apply(lambda x: x['total_tcam_tbl_usage'] / available_TCAM_table, axis=1)
    models_df['success_score'] = models_df.apply(
        lambda x: (0.5 * x['Avg_F1_score'] + 0.5 * (1 - x['total_tcam_usage'])), axis=1)
    return models_df


def select_best_models_per_cluster(cluster_info, score_per_class_df, folder_name):

    n_of_clusters = cluster_info.shape[0]
    cluster_info = cluster_info.assign(Feats_Names=np.nan)
    cluster_info = cluster_info.astype(dtype={"Feats_Names": "object"})

    directory = os.fsencode(folder_name)
    d_frames = defaultdict(list)
    classes = list(chain.from_iterable(cluster_info['Class List'].to_list()))
    pattern = re.compile('[0-9]+')

    for file in os.listdir(directory):
        file_string = file.decode("utf-8")
        path = os.path.join(folder_name, file_string)
        if os.path.isdir(path):
            # skip directories
            continue
        grep_data = pattern.findall(file_string)
        n_point = int(grep_data[0])
        total_n_classes = int(grep_data[1])
        cl = int(grep_data[2])

        model_analysis_for_nth = pd.read_csv(f'{folder_name}/{file_string}', sep=';',
                                             converters=dict.fromkeys(['feats'], literal_converter))
        model_analysis_for_nth['N'] = n_point
        model_analysis_for_nth['Avg_F1_score'] = model_analysis_for_nth.apply(lambda x: (1) * (x['Macro_f1_FL']),
                                                                              axis=1)
        d_frames[cl].append(model_analysis_for_nth)

    ### For each cluster
    for cl in range(0, n_of_clusters):
        models_for_cluster = pd.concat(d_frames[cl])
        models_for_cluster = models_for_cluster.reset_index()

        #### Calculate TOTAL TCAM usage and calculate SUCCESS SCORE
        models_with_tcam_info = calculate_tcam_for_codetables(models_for_cluster)
        models_with_tcam_info = calculate_tcam_for_featable(models_with_tcam_info)
        models_with_tcam_info = total_tcam_usage(models_with_tcam_info)
        ####

        #### ORDER in terms of SUCCESS SCORE and choose the BEST
        chosen_model_tcam_usage = \
            models_with_tcam_info.sort_values('success_score', ascending=0)['total_tcam_usage'].to_list()[0] * 100
        chosen_model_avg_f1_score = \
            models_with_tcam_info.sort_values('success_score', ascending=0)['Avg_F1_score'].to_list()[0] * 100
        chosen_model_index = models_with_tcam_info.sort_values('success_score', ascending=0).head(1).index.to_list()[0]
        ####

        chosen_model = models_with_tcam_info.loc[chosen_model_index]

        #### Store the information of chosen model
        cluster_info.at[cl, 'Depth'] = chosen_model['depth']
        cluster_info.at[cl, 'Tree'] = int(chosen_model['tree'])
        cluster_info.at[cl, 'Feats'] = int(chosen_model['no_feats'])
        cluster_info.at[cl, 'Feats_Names'] = chosen_model['feats']
        cluster_info.at[cl, 'N_Leaves'] = int(chosen_model['N_Leaves'])
        cluster_info.at[cl, 'N'] = int(chosen_model['N'])
        cluster_info.at[cl, 'Macro_f1_FL'] = chosen_model['Macro_f1_FL'] * 100
        cluster_info.at[cl, 'Weighted_f1_FL'] = chosen_model['Weighted_f1_FL'] * 100
        cluster_info.at[cl, 'Micro_f1_FL'] = chosen_model['Micro_f1_FL'] * 100
        cluster_info.at[cl, 'Total_TCAM_Usage'] = chosen_model['total_tcam_usage'] * 100

        cl_report = convert_str_to_dict(chosen_model['cl_report_FL'])
        result_dict = {}
        for d_keys in cl_report.keys():
            if (d_keys != 'accuracy') and (d_keys != 'micro avg'):
                result_dict[d_keys] = cl_report[d_keys]['f1-score']
                if d_keys in classes:
                    df_col = 'Cluster_F1_Score'
                    score_per_class_df.loc[score_per_class_df['class'] == d_keys, df_col] = cl_report[d_keys][
                                                                                                'f1-score'] * 100
                    score_per_class_df.loc[score_per_class_df['class'] == d_keys, 'Cluster'] = cl
        ####

    return cluster_info, score_per_class_df


def calculate_F1_score(cluster_data_file_path, model_analysis_dir_path):
    results_path = f'{model_analysis_dir_path}/perf_results'

    ### Initialize dataframes to use for statistics
    cluster_info = pd.read_csv(cluster_data_file_path,
                               converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))

    classes = list(chain.from_iterable(cluster_info['Class List'].to_list()))
    classes.sort()

    flow_pkt_counts = pd.read_csv(flow_pkts_data_file_path)
    support = flow_pkt_counts['label'].value_counts().loc[classes].sort_index()


    score_per_class_df = pd.DataFrame({'class': classes, 'support': support.to_list()})

    score_per_class_df['Cluster'] = [-1]*len(score_per_class_df)
    # score_per_class_df['Cluster_F1_Score_With_Others'] = [-1]*len(score_per_class_df)


    cluster_info = cluster_info.drop(['Unnamed: 0'], axis=1)
    cluster_info = cluster_info.set_index('Cluster', drop=True)
    cluster_info, score_per_cluster_per_class_df = select_best_models_per_cluster(cluster_info, score_per_class_df,
                                                                                  model_analysis_dir_path)

    #Create folder for saving results
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    cluster_info.to_csv(f'{results_path}/cluster_info_df.csv')
    score_per_cluster_per_class_df.to_csv(f'{results_path}/score_per_cluster_per_class_df.csv')

    ## total TCAM usage
    total_TCAM = sum(cluster_info['Total_TCAM_Usage'].to_list()[1:])

    # score_calc_df = score_per_class_df[score_per_class_df['f1_score']>5]
    score_per_class_df['mult_With_Others'] = score_per_class_df['Cluster_F1_Score']*score_per_class_df['support']
    scores = calculate_score(np.sum(np.array(score_per_class_df['support'])), score_per_class_df['mult_With_Others'],
                             score_per_class_df['Cluster_F1_Score'])
    return scores