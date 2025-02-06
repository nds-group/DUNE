import json
from ast import literal_eval
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import os
import re

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


def calculate_tcam_for_codetables(models_df):
    """ Calculates the required TCAM for each feature table
        :param models_df: the DataFrame describing the partition with the TCAM usage information for code tables
        :return: the DataFrame describing the partition with the TCAM usage information for code tables
    """
    models_df['tcam_codetable'] = models_df.apply(lambda x: (int(x['N_Leaves'] / 44) + 1) * x['tree'], axis=1)
    return models_df


def calculate_tcam_for_featables(models_df):
    """ Calculates the required TCAM for each feature table
    :param models_df: the DataFrame describing the partition with the TCAM usage information for feature tables
    :return: the DataFrame describing the partition with the TCAM usage information for feature tables
    """
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
    """Computes the total TCAM used by the select partition
    :param models_df: the DataFrame describing the partition
    :return: the DataFrame describing the partition with the TOTAL TCAM usage information
    """
    models_df['total_tcam_tbl_usage'] = models_df.apply(lambda x: x['tcam_feature_table'] + x['tcam_codetable'], axis=1)
    models_df['total_tcam_usage'] = models_df.apply(lambda x: x['total_tcam_tbl_usage'] / available_TCAM_table, axis=1)
    return models_df


def select_best_models_per_cluster(cluster_info, analysis_files_dir, support) -> pd.DataFrame:
    """ Selects the best model based on the available model analysis files and the flow pkts data
    :param cluster_info: a Dataframe containing the cluster information, i.e., classes and features information
    :param analysis_files_dir: Path to the analysis files directory
    :param support: a list with the support values matching the classes list
    :return: a DataFrame with the characteristics of each selected model in each row
    """

    n_of_clusters = cluster_info.shape[0]

    directory = os.fsencode(analysis_files_dir)
    d_frames = defaultdict(list)
    classes = list(chain.from_iterable(cluster_info['Class List'].to_list()))
    pattern = re.compile('[0-9]+')

    score_per_class_df = pd.DataFrame({'class': classes, 'support': support.to_list()})
    score_per_class_df['Cluster'] = [-1] * len(score_per_class_df)

    for file in os.listdir(directory):
        # ToDo: filter on the inference point list in the ini file
        file_string = file.decode("utf-8")
        path = os.path.join(directory, file)
        if os.path.isdir(path):
            # skip directories
            continue
        base = os.path.basename(file_string)
        stem = os.path.splitext(base)[0]
        extension = os.path.splitext(base)[1]
        if 'csv' not in extension:
            continue
        if 'solution.csv' in file_string:
            continue # this is the solution file

        grep_data = pattern.findall(file_string)
        n_point = int(grep_data[0])
        cl = int(grep_data[2])

        model_analysis_for_nth = pd.read_csv(f'{analysis_files_dir}/{file_string}', sep=';',
                                             converters=dict.fromkeys(['feats'], literal_converter))
        model_analysis_for_nth['N'] = n_point
        model_analysis_for_nth['Avg_F1_score'] = model_analysis_for_nth.apply(lambda x: (1) * (x['Macro_f1_FL']),
                                                                              axis=1)
        d_frames[cl].append(model_analysis_for_nth)

    chosen_models=[]
    ### For each cluster
    for cl in range(0, n_of_clusters):
        models_for_cluster = pd.concat(d_frames[cl])
        models_for_cluster = models_for_cluster.reset_index(drop=True)

        #### Calculate TOTAL TCAM usage and calculate SUCCESS SCORE
        models_with_tcam_info = calculate_tcam_for_codetables(models_for_cluster)
        models_with_tcam_info = calculate_tcam_for_featables(models_with_tcam_info)
        models_with_tcam_info = total_tcam_usage(models_with_tcam_info)
        ####

        models_with_tcam_info['success_score'] = models_with_tcam_info.apply(
            lambda x: (0.5 * x['Avg_F1_score'] + 0.5 * (1 - x['total_tcam_usage'])), axis=1)

        #### ORDER in terms of SUCCESS SCORE and choose the BEST
        chosen_model_index = models_with_tcam_info.sort_values('success_score', ascending=0).head(1).index.values
        ####
        chosen_models.append(models_with_tcam_info.loc[chosen_model_index])

    best_models_df = pd.concat(chosen_models).reset_index(drop=True)
    best_models_df.index.name = 'Cluster'
    return best_models_df


def append_best_models_info_to_cluster_info(cluster_info: pd.DataFrame, best_models_df: pd.DataFrame) -> pd.DataFrame:
    """Append the information of the best models to the cluster_info DataFrame
    :param cluster_info: a Dataframe containing the cluster information, i.e., classes and features information
    :param best_models_df: a DataFrame with the best models information
    :return: a DataFrame with the information of the best models information
    """
    n_of_clusters = cluster_info.shape[0]

    for cl in range(0, n_of_clusters):
        #### Store the information of chosen model
        cluster_info.at[cl, 'Depth'] = best_models_df.loc[cl]['depth']
        cluster_info.at[cl, 'Tree'] = int(best_models_df.loc[cl]['tree'])
        cluster_info.at[cl, 'Feats'] = int(best_models_df.loc[cl]['no_feats'])
        cluster_info.at[cl, 'Feature List'] = list(best_models_df.loc[cl]['feats'])
        cluster_info.at[cl, 'N_Leaves'] = int(best_models_df.loc[cl]['N_Leaves'])
        cluster_info.at[cl, 'N'] = int(best_models_df.loc[cl]['N'])
        cluster_info.at[cl, 'Macro_f1_FL'] = best_models_df.loc[cl]['Macro_f1_FL'] * 100
        cluster_info.at[cl, 'Weighted_f1_FL'] = best_models_df.loc[cl]['Weighted_f1_FL'] * 100
        cluster_info.at[cl, 'Micro_f1_FL'] = best_models_df.loc[cl]['Micro_f1_FL'] * 100
        cluster_info.at[cl, 'Total_TCAM_Usage'] = best_models_df.loc[cl]['total_tcam_usage'] * 100

    return cluster_info


def generate_score_per_class_report_for_best_models(classes, best_models_df: pd.DataFrame, support) -> pd.DataFrame:
    """Generate a report of the performance scores for each class in the best models DataFrame
    :param classes: a list with all the classes in the cluster
    :param best_models_df: a DataFrame with the best models information
    :param support: a list with the support values matching the classes list
    :return: a DataFrame with the performance scores for each class in the best models DataFrame
    """
    # initialise dataframe
    score_per_class_df = pd.DataFrame({'class': classes, 'support': support.to_list()})
    score_per_class_df['Cluster'] = [-1] * len(score_per_class_df)

    for cluster_id, row in best_models_df.iterrows():
        cl_report = convert_str_to_dict(row['cl_report_FL'])
        result_dict = {}
        for d_keys in cl_report.keys():
            if (d_keys != 'accuracy') and (d_keys != 'micro avg'):
                result_dict[d_keys] = cl_report[d_keys]['f1-score']
                if d_keys in classes:
                    df_col = 'Cluster_F1_Score'
                    score_per_class_df.loc[score_per_class_df['class'] == d_keys, df_col] = cl_report[d_keys][
                                                                                                'f1-score'] * 100
                    score_per_class_df.loc[score_per_class_df['class'] == d_keys, 'Cluster'] = cluster_id
    return score_per_class_df


def calculate_f1_score(score_per_class_df):
    """Compute the F1 score of the provided report
    :param score_per_class_df: a DataFrame with the performance scores report for each class in the best models DataFrame
    :return: a tuple with the average macro and weighted f1 score across all the classes
    """
    score_per_class_df['mult_With_Others'] = score_per_class_df['Cluster_F1_Score']*score_per_class_df['support']

    def calculate_score(support_total, mult_score_support, score):
        macro_score = np.mean(np.array(score))
        weighted_score = np.sum(mult_score_support) / support_total

        return macro_score, weighted_score

    scores = calculate_score(np.sum(np.array(score_per_class_df['support'])), score_per_class_df['mult_With_Others'],
                             score_per_class_df['Cluster_F1_Score'])
    return scores

def calculate_TOTAL_TCAM_usage(cluster_info):
    """Computes the TOTAL TCAM used by the distributed model
    :param cluster_info: a Dataframe containing the cluster information, i.e., classes and features information
    return: the total TCAM usage
    """
    total_TCAM = sum(cluster_info['Total_TCAM_Usage'].to_list()[1:])
    return total_TCAM