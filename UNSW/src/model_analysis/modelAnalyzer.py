import logging
import os
import signal
import shutil
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from os import path
from abc import ABC, abstractmethod

import warnings


def assign_sample_nature(row):
    """Aux function to check the conditions and assign values"""
    if (row["Min Packet Length"] == -1 and
            row["Max Packet Length"] == -1 and
            row["Flow IAT Min"] == -1 and
            row["Flow IAT Max"] == -1):
        return "pkt"
    else:
        return "flw"


class ModelAnalyzer(ABC):
    def __init__(self, train_data_folder_path, test_data_folder_path, classes_filter,
                 cluster_data_file_path, logger):
        self.logger = logger
        self.cluster_data_file_path = cluster_data_file_path
        self.number_of_clusters = pd.read_csv(cluster_data_file_path).shape[0]
        self.train_data_folder_path = train_data_folder_path
        self.test_data_folder_path = test_data_folder_path
        self.classes_filter = classes_filter
        self.feature_list = None
        self.classes = None
        self.classes_df = None
        warnings.filterwarnings("ignore")
        pd.options.mode.chained_assignment = None

    @abstractmethod
    def prepare_data(self, npkts, classes_filter=None):
        pass

    def analyze_models(self, classes, features, depths, n_trees, X_train, y_train, X_test, y_test, samples_nature,
                       y_multiply, test_flow_pkt_cnt, test_flow_IDs, max_leaf, test_labels, test_indices, filename,
                       weight_of_samples, grid_search=False):
        # open file to save ouput of analysis
        root = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]
        tmp_filename = f'{root}_tmp{extension}'
        if os.path.isfile(tmp_filename):
            self.logger.info(f"Deleting existing temp file: {tmp_filename}")
            os.remove(tmp_filename)

        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        def signal_handler(sig, frame):
            self.logger.info('You pressed Ctrl+C! Deleting temporary files...')
            os.remove(tmp_filename)
            raise KeyboardInterrupt()
            # sys.exit(130)

        with open(tmp_filename, "w") as res_file:
            if grid_search:
                self.write_grid_search(X_test, X_train, classes, depths, max_leaf, n_trees, res_file, samples_nature,
                                       signal_handler, test_flow_IDs, test_flow_pkt_cnt, test_indices, test_labels,
                                       tmp_filename, weight_of_samples, y_multiply, y_test, y_train)
            else:
                self.write_simple_analysis(X_test, X_train, classes, features, depths, max_leaf, res_file,
                                           samples_nature,
                                           signal_handler, test_flow_IDs, test_flow_pkt_cnt, test_indices, test_labels,
                                           tmp_filename, weight_of_samples, y_multiply, y_test, y_train)

        shutil.move(tmp_filename, filename)
        logging.getLogger("UNSW").info(f'Finished model analysis. Saved results to: {filename}')
        signal.signal(signal.SIGINT, original_sigint_handler)
        return []

    def write_simple_analysis(self, X_test, X_train, classes, features, depths, max_leaf, res_file, samples_nature,
                              signal_handler, test_flow_IDs, test_flow_pkt_cnt, test_indices, test_labels, tmp_filename,
                              weight_of_samples, y_multiply, y_test, y_train):
        self.logger.info(f'Writing grid search results to: {tmp_filename}')
        print(
            'depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;pkt_macro_f1'
            ';pkt_weighted_f1;flw_macro_f1;flw_weighted_f1;F1_macro;F1_weighted;num_samples;Macro_F1_PL'
            ';Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
            file=res_file)
        # register signal handler to delete file if code is not completed
        signal.signal(signal.SIGINT, signal_handler)
        # for depth in depths:
        depth = depths
        for leaf in max_leaf:
            # Get the scores with the given (depth, n_tree, feat)
            model, c_report, macro_f1, weight_f1, y_pred = get_scores(classes, depth, 1, features,
                                                                      leaf, X_train, y_train, X_test,
                                                                      y_test, test_indices, test_labels,
                                                                      weight_of_samples)
            #
            pkt_macro_f1, pkt_weighted_f1, flw_macro_f1, flw_weighted_f1 = compute_flow_pkt_scores(
                y_pred,
                y_test,
                samples_nature,
                test_indices,
                test_labels)
            #
            num_samples, macro_f1_PL, weighted_f1_PL, micro_f1_PL, cl_report_PL, macro_f1_FL, weighted_f1_FL, micro_f1_FL, cl_report_FL = expand_rows_and_get_scores(
                y_test, y_pred, samples_nature, y_multiply, test_flow_pkt_cnt, test_flow_IDs,
                test_indices,
                test_labels)
            #
            depth = [estimator.tree_.max_depth for estimator in model.estimators_]
            print(str(depth) + ';' + '1' + ';' + str(len(features)) + ';' + str(leaf) + ";" + str(
                macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" + str(
                list(features)) + ';' + str(pkt_macro_f1) + ';' + str(pkt_weighted_f1) + ';' + str(
                flw_macro_f1) + ';' + str(flw_weighted_f1) + ';' + str(macro_f1) + ';' + str(
                weight_f1) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(
                weighted_f1_PL) + ';' + str(micro_f1_PL) + ';' + str(cl_report_FL) + ';' + str(
                cl_report_PL), file=res_file)

    def write_grid_search(self, X_test, X_train, classes, depths, max_leaf, n_trees, res_file, samples_nature,
                          signal_handler, test_flow_IDs, test_flow_pkt_cnt, test_indices, test_labels, tmp_filename,
                          weight_of_samples, y_multiply, y_test, y_train):
        self.logger.info(f'Writing grid search results to: {tmp_filename}')
        print(
            'depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;pkt_macro_f1'
            ';pkt_weighted_f1;flw_macro_f1;flw_weighted_f1;F1_macro;F1_weighted;num_samples;Macro_F1_PL'
            ';Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
            file=res_file)
        # register signal handler to delete file if code is not completed
        signal.signal(signal.SIGINT, signal_handler)
        # FOR EACH (depth, n_tree, feat)
        for n_tree in n_trees:
            # for depth in depths:
            depth = depths
            for leaf in max_leaf:
                # get feature orders to use
                importance = get_feature_importance(depth, n_tree, leaf, X_train, y_train, weight_of_samples)
                m_feats = get_fewest_features(depth, n_tree, leaf, importance)
                for feats in m_feats:
                    # Get the scores with the given (depth, n_tree, feat)
                    model, c_report, macro_f1, weight_f1, y_pred = get_scores(classes, depth, n_tree, feats,
                                                                              leaf, X_train, y_train, X_test,
                                                                              y_test, test_indices, test_labels,
                                                                              weight_of_samples)
                    #
                    pkt_macro_f1, pkt_weighted_f1, flw_macro_f1, flw_weighted_f1 = compute_flow_pkt_scores(
                        y_pred,
                        y_test,
                        samples_nature,
                        test_indices,
                        test_labels)
                    #
                    num_samples, macro_f1_PL, weighted_f1_PL, micro_f1_PL, cl_report_PL, macro_f1_FL, weighted_f1_FL, micro_f1_FL, cl_report_FL = expand_rows_and_get_scores(
                        y_test, y_pred, samples_nature, y_multiply, test_flow_pkt_cnt, test_flow_IDs,
                        test_indices,
                        test_labels)
                    #
                    depth = [estimator.tree_.max_depth for estimator in model.estimators_]
                    print(str(depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(leaf) + ";" + str(
                        macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" + str(
                        list(feats)) + ';' + str(pkt_macro_f1) + ';' + str(pkt_weighted_f1) + ';' + str(
                        flw_macro_f1) + ';' + str(flw_weighted_f1) + ';' + str(macro_f1) + ';' + str(
                        weight_f1) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(
                        weighted_f1_PL) + ';' + str(micro_f1_PL) + ';' + str(cl_report_FL) + ';' + str(
                        cl_report_PL), file=res_file)

    #
    def analyze_model_n_packets(self, npkts, outfile, force=False, grid_search=False):
        """Function for grid search on hyperparameters, features and models"""
        if not force:
            if path.isfile(outfile):
                self.logger.info(
                    f"File {outfile} is present. To overwrite existing files pass force=True when running the"
                    f" analysis")
                return

        # Get Variables and Labels
        train_data, test_data = self.prepare_data(npkts, self.classes_filter)

        test_labels, test_indices = self.get_test_labels(test_data)
        self.logger.debug(f'Num Labels: {len(test_labels)}')

        weight_of_samples = list(train_data['weight'])

        y_multiply = test_data['multiply'].astype(int)
        test_flow_pkt_cnt = test_data['pkt_count'].to_list()
        test_flow_IDs = test_data['Flow ID'].to_list()
        X_train, y_train, sample_nat_train = self.get_x_y_flow(train_data, self.feature_list)
        X_test, y_test, sample_nat_test = self.get_x_y_flow(test_data, self.feature_list)

        depths = []
        trees = [1, 2, 3, 4, 5]
        # Max values per TCAM table, i.e., 85 would require 2 TCAM tables
        val_of_max_leaves = [41, 85, 129, 173, 217, 261, 305, 349, 393, 437, 481, 500]

        #
        self.analyze_models(self.classes, self.feature_list, depths, trees, X_train, y_train, X_test, y_test,
                            sample_nat_test, y_multiply, test_flow_pkt_cnt, test_flow_IDs, val_of_max_leaves,
                            test_labels, test_indices, outfile, weight_of_samples, grid_search)
        self.logger.debug(f'Analysis completed. Check output file: {outfile}')

    def load_cluster_data(self, cluster_data_series):
        # classes_str = str(pd.read_csv(cluster_data_file_path)['Class List'].to_list()[cluster_id])[2:-2]
        # classes = classes_str.split("', '")
        # classes.append('Other')
        classes = list(set(cluster_data_series['Class List']))
        classes.append('Other')
        self.logger.debug(f'Cluster ID: {cluster_data_series["Cluster"]}', classes)
        #
        classes_df = pd.DataFrame(classes, columns=['class'])
        classes_df = classes_df.reset_index()
        #
        # feats_important = pd.read_csv(cluster_data_file_path)['Feature List'].to_list()[cluster_id][2:-2].split("', '")
        feature_list = cluster_data_series['Feature List']
        self.logger.debug(f'Cluster ID: {cluster_data_series["Cluster"]}', feature_list)
        self.feature_list = feature_list
        self.classes = classes
        self.classes_df = classes_df

    @abstractmethod
    def get_test_labels(self, test_data):
        pass

    @abstractmethod
    def get_x_y_flow(self, Dataset, feats):
        pass


class UNSWModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_file_path, classes_filter,
                 cluster_data_file_path, logger):
        super().__init__(train_data_folder_path, test_data_folder_path, classes_filter,
                         cluster_data_file_path, logger)
        self.flow_counts_file_path = flow_counts_file_path

    def get_test_labels(self, test_data):
        array_of_indices = []
        unique_labels = test_data["Label_NEW"].unique()
        for lab in unique_labels:
            index = self.classes_df[self.classes_df['class'] == lab].index.values[0]
            array_of_indices.append(index)
        return unique_labels, array_of_indices

    def get_x_y_flow(self, Dataset, feats):
        X = Dataset[feats]
        y = Dataset['Label_NEW'].replace(self.classes, range(len(self.classes)))
        sample_nature = Dataset['sample_nature']
        return X, y, sample_nature

    def prepare_data(self, npkts, classes_filter=None):
        # Load Train and Test data
        train_data = pd.read_csv(f"{self.train_data_folder_path}/train_data_{npkts}.csv")
        test_data = pd.read_csv(f"{self.test_data_folder_path}/16-10-05.pcap.txt_{npkts}_pkts.csv")

        if classes_filter is not None:
            train_data = train_data.loc[train_data['Label'].isin(classes_filter)]
            test_data = test_data.loc[test_data['Label'].isin(classes_filter)]

        flow_pkt_counts = pd.read_csv(self.flow_counts_file_path)

        flow_count_dict = flow_pkt_counts.set_index("flow.id")["count"].to_dict()
        # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)

        #### To get packet count of each flow in train data
        packet_data = pd.read_csv(f'{self.train_data_folder_path}/UNSW_train_ALL_PKT_DATE.csv')
        packet_data = packet_data[['Flow ID', 'packet_count', 'File']]
        packet_data['File_ID'] = packet_data['Flow ID'] + ' ' + packet_data['File']
        packet_data = packet_data.drop_duplicates(subset='File_ID', keep='first')
        train_data['File_ID'] = train_data['Flow ID'] + ' ' + train_data['File']

        flow_count_dict_train = packet_data.set_index("File_ID")["packet_count"].to_dict()
        # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
        train_data["pkt_count"] = train_data["File_ID"].map(flow_count_dict_train)

        all_minus_one = (test_data['Min Packet Length'] == -1) & (test_data['Max Packet Length'] == -1) & (
                test_data['Packet Length Mean'] == -1)
        # Assign values to the multiply column based on the conditions
        test_data['multiply'] = np.where(all_minus_one, 1, test_data['pkt_count'] - npkts)

        train_data = train_data.sample(frac=1, random_state=42)
        test_data = test_data.sample(frac=1, random_state=42)

        train_data = train_data.dropna(subset=['srcport', 'dstport'])
        test_data = test_data.dropna(subset=['srcport', 'dstport'])

        train_data['Label_NEW'] = np.where((train_data['Label'].isin(self.classes)), train_data['Label'], 'Other')
        test_data['Label_NEW'] = np.where((test_data['Label'].isin(self.classes)), test_data['Label'], 'Other')
        self.logger.debug(f"Train data count: {train_data['Label_NEW'].value_counts()}")
        self.logger.debug(f"Test data count: {test_data['Label_NEW'].value_counts()}")

        train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
        test_data['sample_nature'] = test_data.apply(assign_sample_nature, axis=1)

        train_data['weight'] = np.where(train_data['sample_nature'] == 'flw',
                                        (train_data['pkt_count'] - npkts + 1) / train_data['pkt_count'],
                                        1 / train_data['pkt_count'])
        return train_data, test_data


class TONModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter,
                 cluster_data_file_path, logger):
        super().__init__(train_data_folder_path, test_data_folder_path, classes_filter,
                         cluster_data_file_path, logger)
        self.flow_counts_train_file_path = flow_counts_train_file_path
        self.flow_counts_test_file_path = flow_counts_test_file_path

    def get_test_labels(self, IoT_Test):
        array_of_indices = []
        # unique_labels = IoT_Test["Label_NEW"].unique()
        unique_labels = IoT_Test["Label"].unique()
        for lab in unique_labels:
            index = self.classes_df[self.classes_df['class'] == lab].index.values[0]
            array_of_indices.append(index)
        return unique_labels, array_of_indices

    def get_x_y_flow(self, Dataset, feats):
        X = Dataset[feats]
        # y = Dataset['Label_NEW'].replace(classes, range(len(classes)))
        y = Dataset['Label'].replace(self.classes, range(len(self.classes)))
        sample_nature = Dataset['sample_nature']
        return X, y, sample_nature

    def prepare_data(self, npkts, classes_filter=None):
        train_data = pd.read_csv(f"{self.train_data_folder_path}/train_" + str(npkts) + "_pkts.csv")
        test_data = pd.read_csv(f"{self.test_data_folder_path}/test_" + str(npkts) + "_pkts.csv")

        if classes_filter is not None:
            train_data = train_data.loc[train_data['Label'].isin(classes_filter)]
            test_data = test_data.loc[test_data['Label'].isin(classes_filter)]

        flow_pkt_counts = pd.read_csv(self.flow_counts_test_file_path)
        flow_pkt_counts_train = pd.read_csv(self.flow_counts_train_file_path)

        flow_count_dict_train = flow_pkt_counts_train.set_index("Flow ID")["packet_counts"].to_dict()
        train_data["pkt_count"] = train_data["Flow ID"].map(flow_count_dict_train)
        flow_count_dict = flow_pkt_counts.set_index("Flow ID")["packet_counts"].to_dict()
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)

        flow_count_dict = flow_pkt_counts.set_index("Flow ID")["packet_counts"].to_dict()
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)
        all_minus_one = (test_data['Min Packet Length'] == -1) & (test_data['Max Packet Length'] == -1) & (
                test_data['Packet Length Mean'] == -1)
        # Assign values to the multiply column based on the conditions
        test_data['multiply'] = np.where(all_minus_one, 1, test_data['pkt_count'] - npkts)

        train_data = train_data.sample(frac=1, random_state=42)
        test_data = test_data.sample(frac=1, random_state=42)

        train_data = train_data.dropna(subset=['srcport', 'dstport'])
        test_data = test_data.dropna(subset=['srcport', 'dstport'])

        train_data = train_data[train_data['Label'].isin(self.classes)]
        test_data = test_data[test_data['Label'].isin(self.classes)]

        train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
        test_data['sample_nature'] = test_data.apply(assign_sample_nature, axis=1)

        train_data['weight'] = np.where(train_data['sample_nature'] == 'flw',
                                        (train_data['pkt_count'] - npkts + 1) / train_data['pkt_count'],
                                        1 / train_data['pkt_count'])

        return train_data, test_data


def get_feature_importance(depth, n_tree, max_leaf, X_train, y_train, weight_of_samples):
    """
    Function to Fit model based on optimal values of depth and number of estimators and use it
    to compute feature importance for all the features.
    """
    rf_opt = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_leaf, random_state=42, bootstrap=False,
                                    n_jobs=10)
    rf_opt.fit(X_train, y_train, sample_weight=weight_of_samples)
    feature_importance = pd.DataFrame(rf_opt.feature_importances_)
    feature_importance.index = X_train.columns
    feature_importance = feature_importance.sort_values(by=list(feature_importance.columns), axis=0, ascending=False)

    return feature_importance


def get_fewest_features(depth, n_tree, max_leaf, importance):
    """
    Function to Fit model based on optimal values of depth and number of estimators and feature importance
    to find the fewest possible features to exceed the previously attained score with all selected features
    """
    sorted_feature_names = importance.index
    features = []
    for f in range(1, len(sorted_feature_names) + 1):
        features.append(sorted_feature_names[0:f])
    return features


def get_scores(classes, depth, n_tree, feats, max_leaf, X_train, y_train, X_test, y_test, unique_labels,
               array_of_indices, weight_of_samples):
    model = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_leaf, n_jobs=10,
                                   random_state=42, bootstrap=False)

    model.fit(X_train[feats], y_train, sample_weight=weight_of_samples)
    y_pred = model.predict(X_test[feats])

    y_test = [int(label) for label in y_test.values]
    y_pred = [int(label) for label in y_pred]

    class_report = classification_report(y_test, y_pred, labels=unique_labels, target_names=array_of_indices,
                                         output_dict=True)

    macro_score = class_report['macro avg']['f1-score']
    weighted_score = class_report['weighted avg']['f1-score']

    return model, class_report, macro_score, weighted_score, y_pred


def expand_rows_and_get_scores(y_true, y_pred, sample_nature, multiply, test_flow_pkt_cnt, test_flow_IDs, unique_labels,
                               array_of_indices):
    expanded_y_true = []
    expanded_y_pred = []
    #
    expanded_weights = []
    expanded_flow_IDs = []

    for true_label, pred_label, nature, mult, pkt_cnt, f_id in zip(y_true, y_pred, sample_nature, multiply,
                                                                   test_flow_pkt_cnt, test_flow_IDs):
        if nature == 'flw':
            expanded_y_true.extend([true_label] * (mult + 1))
            expanded_y_pred.extend([pred_label] * (mult + 1))
            #
            expanded_weights.extend([1 / pkt_cnt] * (mult + 1))
            expanded_flow_IDs.extend([f_id] * (mult + 1))
        else:
            expanded_y_true.append(true_label)
            expanded_y_pred.append(pred_label)
            #
            expanded_weights.append(1 / pkt_cnt)
            expanded_flow_IDs.append(f_id)

    num_samples = len(expanded_y_true)

    expanded_y_true = [int(label) for label in expanded_y_true]
    expanded_y_pred = [int(label) for label in expanded_y_pred]
    cl_report_PL = classification_report(expanded_y_true, expanded_y_pred, labels=unique_labels,
                                         target_names=array_of_indices, output_dict=True)
    macro_f1_PL = cl_report_PL['macro avg']['f1-score']
    weighted_f1_PL = cl_report_PL['weighted avg']['f1-score']
    try:
        micro_f1_PL = cl_report_PL['micro avg']['f1-score']
    except:
        micro_f1_PL = cl_report_PL['accuracy']
    ####

    c_report_FL = classification_report(expanded_y_true, expanded_y_pred, labels=unique_labels,
                                        target_names=array_of_indices, output_dict=True, sample_weight=expanded_weights)

    macro_f1_FL = c_report_FL['macro avg']['f1-score']
    weighted_f1_FL = c_report_FL['weighted avg']['f1-score']
    try:
        micro_f1_FL = c_report_FL['micro avg']['f1-score']
    except:
        micro_f1_FL = c_report_FL['accuracy']

    return num_samples, macro_f1_PL, weighted_f1_PL, micro_f1_PL, cl_report_PL, macro_f1_FL, weighted_f1_FL, micro_f1_FL, c_report_FL


def compute_flow_pkt_scores(y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    # Create a data frame with the three columns
    df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test, 'sample_nature': sample_nature})

    # Split the data frame into two data frames based on sample_nature
    pkt_df = df[df['sample_nature'] == 'pkt']
    flw_df = df[df['sample_nature'] == 'flw']

    # Compute macro and weighted F1 scores for pkt_df
    pkt_df_y_true = [int(label) for label in pkt_df['y_test'].values]
    pkt_df_y_pred = [int(label) for label in pkt_df['y_pred']]

    pkt_macro_f1 = \
        classification_report(pkt_df_y_true, pkt_df_y_pred, labels=unique_labels, target_names=array_of_indices,
                              output_dict=True)['macro avg']['f1-score']
    pkt_weighted_f1 = \
        classification_report(pkt_df_y_true, pkt_df_y_pred, labels=unique_labels, target_names=array_of_indices,
                              output_dict=True)['weighted avg']['f1-score']

    # Compute macro and weighted F1 scores for flw_df
    flw_df_y_true = [int(label) for label in flw_df['y_test'].values]
    flw_df_y_pred = [int(label) for label in flw_df['y_pred']]

    flw_macro_f1 = \
        classification_report(flw_df_y_true, flw_df_y_pred, labels=unique_labels, target_names=array_of_indices,
                              output_dict=True)['macro avg']['f1-score']
    flw_weighted_f1 = \
        classification_report(flw_df_y_true, flw_df_y_pred, labels=unique_labels, target_names=array_of_indices,
                              output_dict=True)['weighted avg']['f1-score']

    return pkt_macro_f1, pkt_weighted_f1, flw_macro_f1, flw_weighted_f1
