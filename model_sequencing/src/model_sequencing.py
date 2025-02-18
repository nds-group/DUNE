import collections

import pandas as pd
# import numpy as np
import configparser
# from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, DistanceMetric
# import pyomo.environ as pyo
import numpy as np
import gurobipy 
import logging
# from setup_logger import logger
from TSP import TSP_MTZ_Formulation, get_cluster_seq
#ToDo: remove unnecessary imports numpy, sklearn, pyomo, logger
from ast import literal_eval

def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else literal_eval(val)

def assign_sample_nature(row):
    '''
    Function to assign related values to each packet to specify if it will 
    run packet-level or flow-level classification
    '''
    if (row["Min Packet Length"] == -1 and
        row["Max Packet Length"] == -1 and
        row["Flow IAT Min"] == -1 and
        row["Flow IAT Max"] == -1):
        return "pkt"
    else:
        return "flw"
    

def get_cluster_details(cluster_info_df):
    '''
    Function to get the list of clusters, classes and 
    list of clusters corresponding to the given classes.
    '''
    cluster_list = []
    all_classes_in_clusters = []
    cluster_info_all_classes = []
    cluster_no = 0
    for cl in cluster_info_df['Class List'].to_list():
        classes_str = cl[2:-2]
        classes = classes_str.split("', '")
        cluster_list.append(classes)
        all_classes_in_clusters.extend(classes)
        cluster_info_all_classes.extend([cluster_no]*len(classes))
        cluster_no = cluster_no + 1
    # ToDo: replace cluster_info_all_classes with an individual method that returns the cluster number for a given class.
    # ToDo: all_classes_in_clusters can be returned using the cluster_info_df['Class List'].sum() directly.
    # ToDo: cluster_list can be returned using cluster_info_df['Class List'].to_list()
    return cluster_list, all_classes_in_clusters, cluster_info_all_classes


# ToDo: I suggest removing this function.
def get_final_cluster_info(clusters_best_model_info, logger):
    '''
    Function to get the dataframe that holds the information of clusters sequenced
    '''
    # Create the df
    seq_cluster_info = clusters_best_model_info.copy()
    seq_cluster_info = seq_cluster_info[['Cluster', 'Class List', 'Macro_f1_FL_With_Others']]
    # seq_cluster_info = seq_cluster_info[seq_cluster_info['Cluster'] != -1]
    seq_cluster_info['Average_F1_With_Others'] = seq_cluster_info['Macro_f1_FL_With_Others']
    seq_cluster_info = seq_cluster_info[['Cluster', 'Class List', 'Average_F1_With_Others']]

    cluster_list = seq_cluster_info['Class List'].to_list()
    n_of_classes = []
    for cl in cluster_list:
        cl_list = cl[2:-2].split("', '")
        n_of_classes.append(int(len(cl_list)))
    seq_cluster_info['Number_of_classes'] = n_of_classes
    # 
    df_temp = pd.DataFrame()
    # ToDo: the cluster_seq should be a parameter of the function.
    df_temp['Cluster'] = np.array(clusters_seq)[::-1]
    seq_cluster_info_w_others = df_temp.merge(right=seq_cluster_info,on='Cluster')
    #### Generate Other Class
    ordered_cluster = seq_cluster_info_w_others['Cluster'].to_list()
    ordered_class_list = seq_cluster_info_w_others['Class List'].to_list()
    other_classes = []
    for i in range(0, len(ordered_cluster)):
        other_list = []
        for c_list in ordered_class_list[0:i]:
            other_list.extend(c_list[2:-2].split("', '"))
        other_classes.append(other_list)
    seq_cluster_info_w_others['Other_Classes'] = other_classes
    logger.info(f'Sequenced Clusters information: \n {seq_cluster_info_w_others}')
    return seq_cluster_info_w_others


def get_test_labels_others(IoT_Test, classes_df):
    '''
    Function to get the test labels
    '''
    array_of_indices = []
    unique_labels = IoT_Test["Label_NEW"].unique()
    # unique_labels = IoT_Test["Label"].unique()
    for lab in unique_labels:
        index = classes_df[classes_df['class'] == lab].index.values[0]
        array_of_indices.append(index)
    return unique_labels, array_of_indices

# ToDo: this function does simple things but it obscures the code.
#  In my opinion, these operations should be done directly where needed.
#  Same applies for modelAnalyzer.get_x_y_flow.
#  Both do very similar things but the fact that the code is encapsulated in a function makes it hard to reuse.
def get_x_y_flow_others(Dataset, feats, classes, all_classes_in_clusters):
    '''
    Function to get train and test data for the model
    ''' 
    X = Dataset[feats]
    y = Dataset['Label_NEW'].infer_objects(copy=False).replace(classes, range(len(classes)))
    y_all = Dataset['Label'].infer_objects(copy=False).replace(all_classes_in_clusters, range(len(all_classes_in_clusters)))
    sample_nature = Dataset['sample_nature']
    return X, y, sample_nature, y_all

# ToDo: The name of the function does not map its description.
#  Also, there is a lot of code which is not used in this function (lines 142-159).
#  Please, consider reusing cluster_analysis.extend_test_data_with_flow_level_results.
#  If not possible, try to generalize the function to be used in both cases.
#  In any case, unique_labels, and array_of_indices are not needed.
def expand_rows_and_get_scores_others(y_true, y_pred, y_test_ALL, sample_nature, multiply, test_flow_pkt_cnt, test_flow_IDs, unique_labels, array_of_indices):
    """
    Function to calculate the score of the model in terms of Flow-Level metric and 
    return the classification results of all packets
    """
    expanded_y_true = []
    expanded_y_pred = []
    expanded_y_true_all = []
    #
    expanded_weights = []
    expanded_flow_IDs = []
    
    for true_label, pred_label, y_test_ALL_label, nature, mult, pkt_cnt, f_id in zip(y_true, y_pred, y_test_ALL, sample_nature, multiply, test_flow_pkt_cnt, test_flow_IDs):
        if nature == 'flw':
            expanded_y_true.extend([true_label] * (mult+1))
            expanded_y_pred.extend([pred_label] * (mult+1))
            expanded_y_true_all.extend([y_test_ALL_label] * (mult+1))
            #
            expanded_weights.extend([1/pkt_cnt] * (mult+1))
            expanded_flow_IDs.extend([f_id]* (mult+1))
        else:
            expanded_y_true.append(true_label)
            expanded_y_pred.append(pred_label)
            expanded_y_true_all.append(y_test_ALL_label)
            #
            expanded_weights.append(1/pkt_cnt)
            expanded_flow_IDs.append(f_id)
    
    # num_samples = len(expanded_y_true)

    expanded_y_true = [int(label) for label in expanded_y_true]
    expanded_y_pred = [int(label) for label in expanded_y_pred]
    #
    # cl_report_PL = classification_report(expanded_y_true, expanded_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True)
    # # Packet-level score calculations - return if needed
    # macro_f1_PL = cl_report_PL['macro avg']['f1-score']
    # weighted_f1_PL = cl_report_PL['weighted avg']['f1-score']
    # try:
    #     micro_f1_PL = cl_report_PL['micro avg']['f1-score']
    # except:
    #     micro_f1_PL = cl_report_PL['accuracy']
    # ####
    #
    # c_report_FL =  classification_report(expanded_y_true, expanded_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True,sample_weight=expanded_weights)
    # # Flow-level score calculations - return if needed
    # macro_f1_FL = c_report_FL['macro avg']['f1-score']
    # weighted_f1_FL = c_report_FL['weighted avg']['f1-score']
    # try:
    #     micro_f1_FL = c_report_FL['micro avg']['f1-score']
    # except:
    #     micro_f1_FL = c_report_FL['accuracy']
    
    return expanded_y_true, expanded_y_pred, expanded_weights, expanded_y_true_all


def analyze_model(use_case, classes_filter, npkts, n_tree, max_leaf, feats, classes, classes_df, flow_counts_test_file_path, flow_counts_train_file_path, train_data_dir_path, test_data_dir_path):    
    """
    The function that trains a RandomForestClassifier with the provided data and parameters.
    """
    if(use_case == 'UNSW'):
        # Load Train and Test data
        train_data = pd.read_csv(train_data_dir_path+"/train_data_"+str(npkts)+".csv")
        test_data = pd.read_csv(test_data_dir_path+"/16-10-05.pcap.txt_"+str(npkts)+"_pkts.csv")
        #
        flow_pkt_counts = pd.read_csv(flow_counts_test_file_path)
        #
        ### FIX ###
        flow_count_dict = flow_pkt_counts.set_index("flow.id")["count"].to_dict()
        # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)
        ###########
        
        #### To get packet count of each flow in train data
        packet_data = pd.read_csv("/home/beyzabutun/shared/UNSW_PCAPS/train/train_data_hybrid/UNSW_train_ALL_PKT_DATE.csv")
        packet_data = packet_data[['Flow ID', 'packet_count', 'File']]
        packet_data['File_ID'] = packet_data['Flow ID'] + ' ' + packet_data['File']
        packet_data = packet_data.drop_duplicates(subset='File_ID', keep='first')
        train_data['File_ID'] = train_data['Flow ID'] + ' ' + train_data['File']
        
        flow_count_dict_train = packet_data.set_index("File_ID")["packet_count"].to_dict()
        # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
        train_data["pkt_count"] = train_data["File_ID"].map(flow_count_dict_train)
        ###########
    if(use_case=='TON-IOT'):
        # Load Train and Test data
        train_data = pd.read_csv(train_data_dir_path+"/train_"+str(npkts)+"_pkts.csv")
        test_data = pd.read_csv(test_data_dir_path+"/test_"+str(npkts)+"_pkts.csv")
        #
        flow_pkt_counts = pd.read_csv(flow_counts_test_file_path)
        flow_pkt_counts_train = pd.read_csv(flow_counts_train_file_path)

        flow_count_dict_train = flow_pkt_counts_train.set_index("Flow ID")["packet_counts"].to_dict()
        train_data["pkt_count"] = train_data["Flow ID"].map(flow_count_dict_train)
        
        flow_count_dict = flow_pkt_counts.set_index("Flow ID")["packet_counts"].to_dict()
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)
    

    all_minus_one = (test_data['Min Packet Length'] == -1) & (test_data['Max Packet Length'] == -1) & (test_data['Packet Length Mean'] == -1)
    # Assign values to the multiply column based on the conditions
    test_data['multiply'] = np.where(all_minus_one, 1, test_data['pkt_count'] - npkts)

    train_data = train_data.sample(frac=1, random_state=42)
    test_data  = test_data.sample(frac=1, random_state=42)

    train_data = train_data.dropna(subset=['srcport', 'dstport']) 
    test_data  = test_data.dropna(subset=['srcport', 'dstport'])
    #########
    train_data = train_data[train_data['Label'].isin(classes_filter)]
    test_data = test_data[test_data['Label'].isin(classes_filter)]
    #########
    
    ####
    train_data['Label_NEW'] = np.where((train_data['Label'].isin(classes)), train_data['Label'], 'Other')
    test_data['Label_NEW'] = np.where((test_data['Label'].isin(classes)), test_data['Label'], 'Other')
    ####

    test_labels, test_indices = get_test_labels_others(test_data, classes_df)
    # print("Num Labels: ", len(test_labels))
    # print("Index: ", test_indices)

    train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
    test_data['sample_nature']  = test_data.apply(assign_sample_nature, axis=1)
    
    train_data['weight'] = np.where(train_data['sample_nature'] == 'flw', (train_data['pkt_count'] - npkts + 1)/train_data['pkt_count'], 1/train_data['pkt_count'])
    weight_of_samples = list(train_data['weight'])

    # Get Variables and Labels
    y_multiply = test_data['multiply'].astype(int)
    test_flow_pkt_cnt = test_data['pkt_count'].to_list()
    test_flow_IDs = test_data['Flow ID'].to_list()
    X_train, y_train, sample_nat_train, y_train_ALL = get_x_y_flow_others(train_data, feats, classes, all_classes_in_clusters)
    X_test,  y_test, sample_nat_test, y_test_ALL  = get_x_y_flow_others(test_data, feats, classes, all_classes_in_clusters)
    
    model = RandomForestClassifier(n_estimators = n_tree, max_leaf_nodes=max_leaf, n_jobs=10,
                                        random_state=42, bootstrap=False)
    
    model.fit(X_train[feats], y_train, sample_weight=weight_of_samples)
    y_pred = model.predict(X_test[feats])

    y_test = [int(label) for label in y_test.values]
    y_pred = [int(label) for label in y_pred]
    # ToDo: why is y_test_ALL not converted to ints?
    expanded_y_true, expanded_y_pred, expanded_weights, expanded_y_true_all = expand_rows_and_get_scores_others(y_test, y_pred, y_test_ALL, sample_nat_test, y_multiply, test_flow_pkt_cnt, test_flow_IDs, test_indices, test_labels)
                           
    return expanded_y_true, expanded_y_pred, expanded_weights, expanded_y_true_all
    
# ToDo: this should probably become a function of ModelAnalyzer. Please check if it is possible.
def get_confusion_matrix(use_case, classes_filter, cluster_list, all_classes_in_clusters, clusters_best_model_info, flow_counts_test_file_path, flow_counts_train_file_path, train_data_dir_path, test_data_dir_path):
    """
    Function to get the confusion matrix of the models of clusters
    """
    cm_matrix = pd.DataFrame()
    # ToDo: why do we need a list comprehension in the below assignment?
    cm_matrix['Classes'] = [i for i in all_classes_in_clusters]
    # ToDo: a DataFrame has by default an index. Column `Cluster` is unnecessary.
    cm_matrix_cluster = pd.DataFrame()
    # cm_matrix_cluster['Clusters'] = [i for i in range(0, len(cluster_list))]
    ##

    for cluster_id in range(0, len(cluster_list)):
        npkts = int(clusters_best_model_info.loc[cluster_id]['N_With_Others'])
        n_tree = int(clusters_best_model_info.loc[cluster_id]['Tree_With_Others'])
        max_n_leaves = int(clusters_best_model_info.loc[cluster_id]['N_Leaves_With_Others'])
        feat_names_str = clusters_best_model_info.loc[cluster_id]['Feats_Names_With_Others']
        # ToDo: use ast.literal to parse a string formatted list. See run_cluster_analysis:130 for an example.
        feat_names_str = feat_names_str[2:-2] 
        feat_names = feat_names_str.split("', '")
        classes = cluster_list[cluster_id]
        classes.append('Other')
        # ToDo: once this becomes a function of ModelAnalyzer, you can remove the classes_df.
        #  Since it won't be neeeded in analyze_model.
        classes_df = pd.DataFrame(classes, columns=['class'])

        # ToDo: use dict comprehension instead of for loop. Or use defaultdict from collections.
        cluster_perf_dict = collections.defaultdict(int)
        # cluster_perf_dict = {cl: 0 for cl in range(0, len(cluster_list))}
        # cluster_perf_dict = {}
        # for cl in cm_matrix_cluster['Clusters'].to_list():
        #     cluster_perf_dict[cl] = 0
        #
        logging.getLogger(f'{use_case}').info(f'The model info: \n {npkts, n_tree, max_n_leaves, feat_names, classes}')
        #
        # ToDo: This function receives too many arguments and returns too many values.
        #  The name of the function should also be more descriptive (should hint what you expect to return).
        #  Since we already analysed all models in the previous step, can we avoid re-training in the future?
        expanded_y_true, expanded_y_pred, expanded_weights, expanded_y_true_all = analyze_model(use_case, classes_filter, npkts, n_tree, max_n_leaves, feat_names, classes, classes_df, flow_counts_test_file_path, flow_counts_train_file_path, train_data_dir_path, test_data_dir_path)
        # ToDo: you can build a Df as pd.Dataframe({'True_Label_Cluster': expanded_y_true, 'Pred_Label_Cluster': expanded_y_pred, 'True_Label_All': expanded_y_true_all, 'Weight_per_packet': expanded_weights})
        pred_df = pd.DataFrame()
        pred_df['True_Label_Cluster'] = expanded_y_true
        pred_df['Pred_Label_Cluster'] = expanded_y_pred
        pred_df['True_Label_All'] = expanded_y_true_all
        pred_df['Weight_per_packet'] = expanded_weights
        #

        # Precompute indices for classes to avoid repeated .index() calls
        class_indices = {cls: idx for idx, cls in enumerate(classes)}
        all_classes_in_clusters_indices = {cla: idx for idx, cla in enumerate(all_classes_in_clusters)}

        # Why do we iterate over the classes in the cluster except the last?
        # Because the last one is the 'Other' class...
        for _class in classes[:-1]:
            performance_vals = []
            # ToDo:filter out already classes not in cluster
            classes_in_cluster_df = pred_df.loc[pred_df['Pred_Label_Cluster'] == classes.index(_class)]
            for cla in all_classes_in_clusters:
                if cla in classes:
                    cla_ind = class_indices[cla]
                    label_column = 'True_Label_Cluster'
                    cluster_of_cls = cluster_id
                    # # metric_val = sum(pred_df[((pred_df['True_Label_Cluster'] == cla_ind) & (pred_df['Pred_Label_Cluster'] == cla_in_clu_ind))]['Weight_per_packet'].to_list())
                    # metric_val = classes_in_cluster_df.loc[
                    #                 (pred_df['True_Label_Cluster'] == cla_ind),
                    #                 'Weight_per_packet'
                    # ].sum()
                    # performance_vals.append(metric_val)
                    # cluster_perf_dict[cluster_of_cls] += metric_val
                else:
                    cla_ind = all_classes_in_clusters_indices[cla]
                    label_column = 'True_Label_All'
                    cluster_of_cls = cluster_info_all_classes[cla_ind]

                # ToDo: Chaining conditions creates long lines, which are discouraged by PEP8. Some alternatives:
                #  - metric_val = pred_df.loc[
                #                 (pred_df['True_Label_Cluster'] == cla_ind) &
                #                 (pred_df['Pred_Label_Cluster'] == cla_in_clu_ind),
                #                 'Weight_per_packet'
                #    ].sum()
                #  - pred_df.query('True_Label_Cluster == 0 & Pred_Label_Cluster == 0')['Weight_per_packet'].sum()
                #  - cond1 = pred_df.True_Label_Cluster == cla_ind
                #    cond2 = pred_df.Pred_Label_Cluster == cla_in_clu_ind
                #    pred_df[cond1 & cond2]['Weight_per_packet'].sum()
                #  Also, if you have a DataFrame object, try to use its method (.sum()) instead of .to_list() and sum() which are not implemented in pandas.
                # metric_val = sum(pred_df[((pred_df['True_Label_All'] == cla_ind) & (pred_df['Pred_Label_Cluster'] == cla_in_clu_ind))]['Weight_per_packet'].to_list())
                metric_val = classes_in_cluster_df.loc[
                    (pred_df[label_column] == cla_ind),
                    'Weight_per_packet'
                ].sum()
                performance_vals.append(metric_val)
                cluster_perf_dict[cluster_of_cls] +=  metric_val

            # ToDo: This assignment can be done once after the loop. See that the  dict is initialized outside the loop.
            #  In general, it is best to find the least amount of code that achieves your goal.
            # cm_matrix_cluster[str(i)] = cluster_perf_dict.values()

            cm_matrix[_class] = performance_vals
        cm_matrix_cluster[str(cluster_id)] = cluster_perf_dict.values()
    # cm_matrix_cluster = pd.DataFrame(cluster_perf_dict.items())


    return cm_matrix, cm_matrix_cluster


# ToDo: remove n_of_clusters. It is not used.
def normalize_confusion_matrix(cm_matrix_cluster, n_of_clusters):
    """
    Function to normalize the confusion matrix
    """
    # ToDo: remove the code below. There is an efficient way to normalize with Pandas.
    # cm_matrix_cluster_normalized_df = pd.DataFrame()
    # cm_matrix_cluster_normalized_df['Clusters'] = cm_matrix_cluster['Clusters'].to_list()
    # cm_matrix_cluster_normalized_df['Clusters'] = cm_matrix_cluster.index.values.tolist()

    # cm_matrix_cluster['sum'] = cm_matrix_cluster.sum(axis=1)

    # cm_matrix_cluster['sum'] = 0
    # cluster_list_str = []
    # for i in range(0, n_of_clusters):
    #     cm_matrix_cluster['sum'] = cm_matrix_cluster['sum'] + cm_matrix_cluster[str(i)]
    #     cluster_list_str.append(str(i))

    # for i in range(0, len(cm_matrix_cluster['Clusters'].to_list())):
    #     cm_matrix_cluster_normalized_df[str(i)] = (cm_matrix_cluster[str(i)] * 100) / sum(
    #         cm_matrix_cluster[str(i)].to_list())

    cm_matrix_cluster_normalized_df = cm_matrix_cluster.div(cm_matrix_cluster.sum(axis=0), axis=1).mul(100)
    
    # return cm_matrix_cluster_normalized_df, cm_matrix_cluster.index.values
    return cm_matrix_cluster_normalized_df


if __name__ == '__main__':
    logging.basicConfig()
    # Read the data
    config = configparser.ConfigParser()
    config.read('params.ini')
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)
    
    # Parameters necessary for the model sequencing
    flow_counts_train_file_path = config[use_case]['flow_counts_train_file_path'] 
    classes_filter = config[use_case]['classes_filter'] 
    classes_filter = classes_filter[2:-2].split("', '")
    flow_counts_test_file_path = config[use_case]['flow_counts_test_file_path'] 
    best_models_path = config[use_case]['best_models_per_cluster_path']
    train_data_dir_path = config[use_case]['train_data_dir_path']
    test_data_dir_path = config[use_case]['test_data_dir_path']
    results_dir_path = config[use_case]['results_dir_path']
    
    # Get the cluster information which is obtained in the previous steps
    # ToDo: use converters to parse list representations.
    #  E.g. pass converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter) to pd.read_csv
    clusters_best_model_info = pd.read_csv(best_models_path)
    # clusters_best_model_info = clusters_best_model_info.drop(['Unnamed: 0'], axis=1)
    # clusters_best_model_info = clusters_best_model_info.set_index('Cluster', drop=True)
    cluster_list, all_classes_in_clusters, cluster_info_all_classes = get_cluster_details(clusters_best_model_info)
    
    # Get the confusion matrix between the models of the corresponding clusters
    logger.info(f'The analysis starts...')
    try:
        cm_matrix = pd.read_csv(results_dir_path + '/' + use_case+'_confusion_matrix.csv')
        cm_matrix_cluster = pd.read_csv(results_dir_path + '/' + use_case + '_cm_matrix_cluster.csv')
    except FileNotFoundError:
        cm_matrix, cm_matrix_cluster = get_confusion_matrix(use_case, classes_filter, cluster_list, all_classes_in_clusters, clusters_best_model_info, flow_counts_test_file_path, flow_counts_train_file_path, train_data_dir_path, test_data_dir_path)
        cm_matrix.to_csv(results_dir_path + '/' + use_case+'_confusion_matrix.csv', index=False)
        cm_matrix_cluster.to_csv(results_dir_path + '/' + use_case + '_cm_matrix_cluster.csv', index=False)
    # ToDo: remove the normalize_confusion_matrix. It is sufficient with a single pandas line. See the function for more details.
    # cm_matrix_cluster_normalized_df, cluster_list_str = normalize_confusion_matrix(cm_matrix_cluster, len(cluster_list))
    cm_matrix_cluster_normalized_df = normalize_confusion_matrix(cm_matrix_cluster, len(cluster_list))
    logger.info(f'Normalized confusion matrix is: ')
    logger.info(f'{cm_matrix_cluster_normalized_df}')
    
    # cost_FP = cm_matrix_cluster_normalized_df[cluster_list_str].to_numpy()
    cost_FP = cm_matrix_cluster_normalized_df.values
    # ToDo: if the above can be a ndarray, then the below as well.
    # cost_F1 = clusters_best_model_info['Macro_f1_FL_With_Others'].to_list()
    cost_F1 = clusters_best_model_info['Macro_f1_FL_With_Others'].values
    # Order the clusters based on the FP, and F1 scores
    # ToDo: only calls to class constructors should be capitalized.
    # The number of clusters can be derived from the cost_FP matrix.
    # solutionObjective, solutionGap, tourRepo, completeResults = TSP_MTZ_Formulation(len(cluster_list), cost_FP, cost_F1, logger)
    solutionObjective, solutionGap, tourRepo, completeResults = TSP_MTZ_Formulation(cost_FP, cost_F1, logger)
    clusters_seq = get_cluster_seq(tourRepo, len(cluster_list), logger)
    # clusters_seq = (2, 0, 1, 3, 4, 5) #UNSW
    
    # Get final clustering information after sequencing
    # ToDo: I suggest to remove the code below, as this code only needs to order.
    seq_cluster_info_w_others = get_final_cluster_info(clusters_best_model_info, logger)
    seq_cluster_info_w_others.to_csv(results_dir_path + use_case+'_sequenced_clusters_info.csv')
