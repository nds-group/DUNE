import os
import signal
import shutil

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from os import path
from abc import ABC, abstractmethod
from itertools import compress

import warnings

# list of all extracted features
feats_all = ['ip.len', 'ip.ttl', 'tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.fin',
             'tcp.flags.rst', 'tcp.flags.ece', 'ip.proto', 'srcport', 'dstport', 'ip.hdr_len', 'tcp.window_size_value',
             'tcp.hdr_len', 'udp.length', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
             'Packet Length Total', 'UDP Len Min', 'UDP Len Max', 'Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean',
             'Flow Duration', 'SYN Flag Count', 'ACK Flag Count', 'PSH Flag Count', 'FIN Flag Count', 'RST Flag Count',
             'ECE Flag Count']



def assign_sample_nature(row):
    """Aux function to check the conditions and assign values"""
    if (row["Min Packet Length"] == -1 and
            row["Max Packet Length"] == -1 and
            row["Flow IAT Min"] == -1 and
            row["Flow IAT Max"] == -1):
        return "pkt"
    else:
        return "flw"


def extend_test_data_with_flow_level_results(y_test, y_pred, samples_nature_test, y_multiply_test, flow_pkt_cnt_test, flow_ids_test):
    expanded_y_test = []
    expanded_y_pred = []
    expanded_weights = []
    expanded_flow_IDs = []
    # If we have 10 packets in the flow and do inference at packet 3, we get multiply = 10 - 3 = 7.
    # We add 1 to include the n-th packet which is where we make inference.
    # This will mean that we want to attribute the score of the n-th packet to the remaining 7th packet but
    # since we want the n-th packet itself to be included in the list of classified packets
    # (this list was empty from the start), we add 1 to include it in the number of packets that take the
    # classification result.

    for true_label, pred_label, nature, multiplier, pkt_cnt, f_id in zip(y_test, y_pred, samples_nature_test,
                                                                         y_multiply_test,
                                                                         flow_pkt_cnt_test, flow_ids_test):
        if nature == 'flw':
            expanded_y_test.extend([true_label] * (multiplier + 1)) # +1 to account for the nth pkt (see above)
            expanded_y_pred.extend([pred_label] * (multiplier + 1))
            #
            expanded_weights.extend([1 / pkt_cnt] * (multiplier + 1))
            expanded_flow_IDs.extend([f_id] * (multiplier + 1))
        else:
            expanded_y_test.append(true_label)
            expanded_y_pred.append(pred_label)
            #
            expanded_weights.append(1 / pkt_cnt)
            expanded_flow_IDs.append(f_id)

    return expanded_y_test, expanded_y_pred, expanded_weights, expanded_flow_IDs



class ModelAnalyzer(ABC):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path, 
                 flow_counts_test_file_path, classes_filter, cluster_data_file_path, logger):
        self.logger = logger
        self.cluster_data_file_path = cluster_data_file_path
        self.train_data_folder_path = train_data_folder_path
        self.test_data_folder_path = test_data_folder_path
        self.flow_counts_train_file_path = flow_counts_train_file_path
        self.flow_counts_test_file_path = flow_counts_test_file_path
        self.classes_filter = classes_filter
        self.feature_list = None
        self.classes = None
        self.classes_df = None
        warnings.filterwarnings("ignore")
        pd.options.mode.chained_assignment = None

    @abstractmethod
    def prepare_data(self, npkts, classes_filter=None):
        pass

    def _prepare_data(self, npkts, classes_filter, train_file, test_file, flow_counts_train, flow_counts_test):
        """
        Generalized method to handle repeated logic in prepare_data for UNSW and TON analyzers.
        """
        # Load Train and Test data
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)

        # Apply class filters if provided
        if classes_filter is not None:
            train_data = train_data.loc[train_data['Label'].isin(classes_filter)]
            test_data = test_data.loc[test_data['Label'].isin(classes_filter)]

        # ToDo: this is specific and should somehow go into the child classes implementation. How to best do it?
        if 'File' in train_data.columns:
            train_data["Flow ID"] = train_data['Flow ID'] + ' ' + train_data['File']
        # Map flow packet counts for train and test sets
        train_data["pkt_count"] = train_data["Flow ID"].map(flow_counts_train)
        # For UNSW this is fine because no multiple files exist, i.e., the test data comes from a single file.
        test_data["pkt_count"] = test_data["Flow ID"].map(flow_counts_test)

        # Shuffle and clean data
        train_data = train_data.sample(frac=1, random_state=42).dropna(subset=['srcport', 'dstport'])
        test_data = test_data.sample(frac=1, random_state=42).dropna(subset=['srcport', 'dstport'])

        # Update labels for 'Label_NEW' column
        train_data['Label_NEW'] = np.where((train_data['Label'].isin(self.classes)), train_data['Label'], 'Other')
        test_data['Label_NEW'] = np.where((test_data['Label'].isin(self.classes)), test_data['Label'], 'Other')

        # Debug info
        self.logger.debug(f"Train data count: {train_data['Label_NEW'].value_counts()}")
        self.logger.debug(f"Test data count: {test_data['Label_NEW'].value_counts()}")

        # Assign 'sample_nature' and 'weight' columns
        train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
        test_data['sample_nature'] = test_data.apply(assign_sample_nature, axis=1)

        # Assign 'multiply' column based on the conditions.
        test_data['multiply'] = np.where(test_data['sample_nature']=='pkt', 1, test_data['pkt_count'] - npkts)

        train_data['weight'] = np.where(train_data['sample_nature'] == 'flw',
                                        (train_data['pkt_count'] - npkts + 1) / train_data['pkt_count'],
                                        1 / train_data['pkt_count'])
        return train_data, test_data

    def analyze_models(self, n_trees, x_train, y_train, x_test, y_test, samples_nature, y_multiply, test_flow_pkt_cnt,
                       test_flow_ids, max_leaf, test_labels, test_indices, filename, weight_of_samples,
                       grid_search=False):
        # open file to save output of analysis
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

        if grid_search:
            self.write_grid_search(x_test, x_train, max_leaf, n_trees, samples_nature, signal_handler, test_flow_ids,
                                   test_flow_pkt_cnt, test_indices, test_labels, tmp_filename, weight_of_samples,
                                   y_multiply, y_test, y_train)
        else:
            self.write_simple_analysis(x_test, x_train, max_leaf, samples_nature, signal_handler, test_flow_ids,
                                       test_flow_pkt_cnt, test_indices, test_labels, tmp_filename, weight_of_samples,
                                       y_multiply, y_test, y_train)

        shutil.move(tmp_filename, filename)
        self.logger.info(f'Finished model analysis. Saved results to: {filename}')
        signal.signal(signal.SIGINT, original_sigint_handler)
        return []

    def write_simple_analysis(self, x_test, x_train, max_leaf, samples_nature_test, signal_handler, flow_ids_test,
                              flow_pkt_cnt_test, test_labels, test_label_names, tmp_filename, weight_of_samples,
                              y_multiply_test, y_test, y_train):
        with open(tmp_filename, "w") as res_file:
            self.logger.info(f'Writing grid search results to: {tmp_filename}')
            print(
                'depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;pkt_macro_f1'
                ';pkt_weighted_f1;flw_macro_f1;flw_weighted_f1;F1_macro;F1_weighted;num_samples;Macro_F1_PL'
                ';Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
                file=res_file)
            # register signal handler to delete file if code is not completed
            signal.signal(signal.SIGINT, signal_handler)
            n_tree =1
            feats = x_train.columns.values.tolist()
            for leaf in max_leaf:
                # Prepare a model for the given (depth, n_tree, feat)
                model = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=leaf, n_jobs=10, random_state=42,
                                               bootstrap=False)
                # Train (fit) the model with the data
                model.fit(x_train[feats], y_train, sample_weight=weight_of_samples)
                # Infer (predict) the labels
                y_pred = model.predict(x_test[feats]).tolist()

                # Obtain a generic classification report. We later drill down to flow and pkt level reports.
                overall_class_report = classification_report(y_test, y_pred, labels=test_labels,
                                                             target_names=test_label_names, output_dict=True)

                overall_macro_f1 = overall_class_report['macro avg']['f1-score']
                overall_weighted_f1 = overall_class_report['weighted avg']['f1-score']

                pkt_class_report = get_pkt_class_report(y_pred, y_test, samples_nature_test, test_labels, test_label_names)
                pkt_macro_f1 = pkt_class_report['macro avg']['f1-score']
                pkt_weighted_f1 = pkt_class_report['weighted avg']['f1-score']

                flow_class_report = get_flow_class_report(y_pred, y_test, samples_nature_test, test_labels,
                                                          test_label_names)
                flw_macro_f1 = flow_class_report['macro avg']['f1-score']
                flw_weighted_f1 = flow_class_report['weighted avg']['f1-score']

                (expanded_y_test,
                 expanded_y_pred,
                 expanded_weights,
                 expanded_flow_IDs) = extend_test_data_with_flow_level_results(y_test,
                                                                               y_pred,
                                                                               samples_nature_test,
                                                                               y_multiply_test,
                                                                               flow_pkt_cnt_test,
                                                                               flow_ids_test)
                num_samples = len(expanded_y_test)

                FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_labels,
                                                        target_names=test_label_names, output_dict=True,
                                                        sample_weight=expanded_weights)

                macro_f1_FL = FL_class_report['macro avg']['f1-score']
                weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                micro_f1_FL = FL_class_report['accuracy']

                PL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_labels,
                                                        target_names=test_label_names, output_dict=True)

                macro_f1_PL = PL_class_report['macro avg']['f1-score']
                weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                micro_f1_PL = PL_class_report['accuracy']

                depth = [estimator.tree_.max_depth for estimator in model.estimators_]
                print(str(depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(leaf) + ";" + str(
                    macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" + str(feats)
                      + ';' + str(pkt_macro_f1) + ';' + str(pkt_weighted_f1) + ';' + str(
                    flw_macro_f1) + ';' + str(flw_weighted_f1) + ';' + str(overall_macro_f1) + ';' + str(
                    overall_weighted_f1) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(
                    weighted_f1_PL) + ';' + str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                      file=res_file)


    def write_grid_search(self, x_test, x_train, max_leaf, n_trees, samples_nature_test, signal_handler, flow_ids_test,
                          flow_pkt_cnt_test, test_labels, test_label_names, tmp_filename, weight_of_samples,
                          y_multiply_test, y_test, y_train):
        with open(tmp_filename, "w") as res_file:
            self.logger.info(f'Writing grid search results to: {tmp_filename}')
            print(
                'depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;pkt_macro_f1'
                ';pkt_weighted_f1;flw_macro_f1;flw_weighted_f1;F1_macro;F1_weighted;num_samples;Macro_F1_PL'
                ';Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
                file=res_file)
            # register signal handler to delete file if code is not completed
            signal.signal(signal.SIGINT, signal_handler)
            # FOR EACH (n_tree, leaf, feat)
            # ToDo: make n_trees a global Int const
            for n_tree in n_trees:
                # ToDo: make max_leaf a global list const
                for leaf in max_leaf:
                    # get feature orders to use
                    m_feats = get_feature_importance_sets(n_tree, leaf, x_train, y_train, weight_of_samples)
                    for feats in m_feats:
                        # ToDo: extract method analyse_grid_point to share code with write_simple_analysis
                        # Prepare a model for the given (depth, n_tree, feat)
                        model = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=leaf, n_jobs=10,
                                                       random_state=42, bootstrap=False)
                        # Train (fit) the model with the data
                        model.fit(x_train[feats], y_train, sample_weight=weight_of_samples)
                        # Infer (predict) the labels
                        y_pred = model.predict(x_test[feats]).tolist()

                        #Obtain a generic classification report. We later drill down to flow and pkt level reports.
                        overall_class_report = classification_report(y_test, y_pred, labels=test_labels,
                                                                     target_names=test_label_names, output_dict=True)

                        overall_macro_f1 = overall_class_report['macro avg']['f1-score']
                        overall_weighted_f1 = overall_class_report['weighted avg']['f1-score']

                        pkt_class_report = get_pkt_class_report(y_pred, y_test, samples_nature_test, test_labels, test_label_names)
                        pkt_macro_f1 = pkt_class_report['macro avg']['f1-score']
                        pkt_weighted_f1 = pkt_class_report['weighted avg']['f1-score']

                        flow_class_report = get_flow_class_report(y_pred, y_test, samples_nature_test, test_labels, test_label_names)
                        flw_macro_f1 = flow_class_report['macro avg']['f1-score']
                        flw_weighted_f1 = flow_class_report['weighted avg']['f1-score']

                        # The result of the flow level inference must be applied to the consecutive packets.
                        # ToDo: use true or test consistently to avoid confusions.
                        #  Be consistent with the naming in the paper.
                        (expanded_y_test,
                         expanded_y_pred,
                         expanded_weights,
                         expanded_flow_IDs) = extend_test_data_with_flow_level_results(y_test,
                                                                                       y_pred,
                                                                                       samples_nature_test,
                                                                                       y_multiply_test,
                                                                                       flow_pkt_cnt_test,
                                                                                       flow_ids_test)
                        num_samples = len(expanded_y_test)

                        FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_labels,
                                                                target_names=test_label_names, output_dict=True,
                                                                sample_weight=expanded_weights)

                        macro_f1_FL = FL_class_report['macro avg']['f1-score']
                        weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                        # try:
                        #     micro_f1_FL = FL_class_report['micro avg']['f1-score']
                        # except:
                        # ToDo: check why was the try/except block needed. Remove above-commented code otherwise.
                        micro_f1_FL = FL_class_report['accuracy']

                        PL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_labels,
                                                                target_names=test_label_names, output_dict=True)

                        macro_f1_PL = PL_class_report['macro avg']['f1-score']
                        weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                        # try:
                        #     micro_f1_PL = PL_class_report['micro avg']['f1-score']
                        # except:
                        # ToDo: check why was the try/except block needed. Remove above-commented code otherwise.
                        micro_f1_PL = PL_class_report['accuracy']

                        depth = [estimator.tree_.max_depth for estimator in model.estimators_]
                        # The metric on which we select the best model is then the macro_f1_FL
                        # ToDo: evaluate model while searching, and keep in memory the best found so far. Save the others as CSV files just in case.
                        print(str(depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' + str(leaf) + ";" + str(
                              macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" + str(feats)
                              + ';' + str(pkt_macro_f1) + ';' + str(pkt_weighted_f1) + ';' + str(
                              flw_macro_f1) + ';' + str(flw_weighted_f1) + ';' + str(overall_macro_f1) + ';' + str(
                              overall_weighted_f1) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(
                              weighted_f1_PL) + ';' + str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                              file=res_file)

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

        # TODO: drop intermediate vars and pass test/train data vars to analyze_models method
        y_multiply = test_data['multiply']
        test_flow_pkt_cnt = test_data['pkt_count'].to_list()
        test_flow_IDs = test_data['Flow ID'].to_list()
        x_train, y_train, sample_nat_train = self.get_x_y_flow(train_data)
        x_test, y_test, sample_nat_test = self.get_x_y_flow(test_data)

        trees = [1, 2, 3, 4, 5]
        # Max values per TCAM table, i.e., 85 would require 2 TCAM tables
        val_of_max_leaves = [41, 85, 129, 173, 217, 261, 305, 349, 393, 437, 481, 500]

        #
        self.analyze_models(trees, x_train, y_train, x_test, y_test, sample_nat_test, y_multiply, test_flow_pkt_cnt,
                            test_flow_IDs, val_of_max_leaves, test_labels, test_indices, outfile, weight_of_samples,
                            grid_search)
        self.logger.debug(f'Analysis completed. Check output file: {outfile}')

    def load_cluster_data(self, cluster_data_series):
        classes = list(set(cluster_data_series['Class List']))
        classes.append('Other')
        self.logger.debug(f'Cluster ID: {cluster_data_series["Cluster"]}', classes)

        classes_df = pd.DataFrame(classes, columns=['class'])
        classes_df = classes_df.reset_index()

        feature_list = cluster_data_series['Feature List']
        self.logger.debug(f'Cluster ID: {cluster_data_series["Cluster"]}', feature_list)
        self.feature_list = feature_list
        self.classes = classes
        self.classes_df = classes_df
    
    def get_test_labels(self, test_data):
        array_of_indices = []
        unique_labels = test_data["Label_NEW"].unique()
        for lab in unique_labels:
            index = self.classes_df[self.classes_df['class'] == lab].index.values[0]
            array_of_indices.append(index)
        return unique_labels, array_of_indices

    def get_x_y_flow(self, dataset, feats=None):
        if feats is None:
            x = dataset[feats_all]
        else:
            x = dataset[feats]

        y = dataset['Label_NEW'].replace(self.classes, range(len(self.classes))).values.tolist()
        sample_nature = dataset['sample_nature']
        return x, y, sample_nature


class UNSWModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter,
                 cluster_data_file_path, logger):
        super().__init__(train_data_folder_path, test_data_folder_path, flow_counts_train_file_path, 
                         flow_counts_test_file_path, classes_filter, cluster_data_file_path, logger)

    def prepare_data(self, npkts, classes_filter=None):
        # Load train and test flow count files
        usecols = ['Flow ID', 'packet_count', 'File']
        flow_counts_train = pd.read_csv(self.flow_counts_train_file_path, usecols=usecols)
        flow_counts_test = pd.read_csv(self.flow_counts_test_file_path)

        # Prepare flow packet counts mapping dictionaries
        flow_count_dict_test = flow_counts_test.set_index("flow.id")["count"].to_dict()
        flow_counts_train['Flow ID'] = flow_counts_train['Flow ID'] + ' ' + flow_counts_train['File']
        flow_counts_train = flow_counts_train.drop_duplicates(subset=['Flow ID'], keep='first')
        flow_count_dict_train = flow_counts_train.set_index("Flow ID")["packet_count"].to_dict()

        # Prepare train and test file paths
        train_file = f"{self.train_data_folder_path}/train_data_{npkts}.csv"
        test_file = f"{self.test_data_folder_path}/16-10-05.pcap.txt_{npkts}_pkts.csv"

        # Call base method to perform common steps
        train_data, test_data = self._prepare_data(npkts, classes_filter, train_file, test_file, flow_count_dict_train,
                                                   flow_count_dict_test)
        return train_data, test_data


class TONModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter,
                 cluster_data_file_path, logger):
        super().__init__(train_data_folder_path, test_data_folder_path, 
                         flow_counts_train_file_path, flow_counts_test_file_path,classes_filter,
                         cluster_data_file_path, logger)

    def prepare_data(self, npkts, classes_filter=None):
        # Load train and test flow count files
        flow_counts_train = pd.read_csv(self.flow_counts_train_file_path)
        flow_counts_test = pd.read_csv(self.flow_counts_test_file_path)

        # Prepare flow packet counts mapping dictionaries
        flow_count_dict_train = flow_counts_train.set_index("Flow ID")["packet_counts"].to_dict()
        flow_count_dict_test = flow_counts_test.set_index("Flow ID")["packet_counts"].to_dict()

        # Prepare train and test file paths
        train_file = f"{self.train_data_folder_path}/train_{npkts}_pkts.csv"
        test_file = f"{self.test_data_folder_path}/test_{npkts}_pkts.csv"

        # Call base method to perform common steps
        train_data, test_data = self._prepare_data(npkts, classes_filter, train_file, test_file,
                                                   flow_count_dict_train, flow_count_dict_test)
        return train_data, test_data


def get_feature_importance_sets(n_tree, max_leaf, x_train, y_train, weight_of_samples):
    """
        Generate feature importance sets in descending order using a trained Random Forest model.

        This function trains a RandomForestClassifier with the provided data and parameters,
        extracts the feature importances, and returns the feature names sorted by their
        importance in descending order. It produces cumulative subsets of feature names
        such that each set includes the most important features up to that rank.

        Arguments:
        n_tree: int
            Number of trees in the RandomForestClassifier.
        max_leaf: int
            Maximum number of leaf nodes for each tree in the RandomForestClassifier.
        x_train: np.ndarray
            Training features, where each row represents a sample and each column a feature.
        y_train: np.ndarray
            Training labels corresponding to each sample in the training data.
        weight_of_samples: np.ndarray
            Sample weights to account for unequal importance or representativity of samples.

        Returns:
        List[List[str]]
            A list of lists of strings, where each nested list contains a cumulative subset of feature names sorted by importance.
    """
    # ToDo: check that the importance analysis evaluates all existing features and not only the features set provided by the SPP.
    rf_opt = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_leaf, random_state=42, bootstrap=False,
                                    n_jobs=10)
    rf_opt.fit(x_train, y_train, sample_weight=weight_of_samples)

    # Get the indices that would sort the feature_importances_ array in descending order
    sorted_indices = np.argsort(rf_opt.feature_importances_)[::-1]

    # Sort both arrays using the sorted indices
    sorted_feature_importances = rf_opt.feature_importances_[sorted_indices]
    sorted_feature_names = rf_opt.feature_names_in_[sorted_indices].tolist()

    return [sorted_feature_names[:i] for i in range(1, len(sorted_feature_names) + 1)]


def get_pkt_class_report(y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    return _compute_scores_by_nature("pkt", y_pred, y_test, sample_nature, unique_labels, array_of_indices)


def get_flow_class_report(y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    return _compute_scores_by_nature("pkt", y_pred, y_test, sample_nature, unique_labels, array_of_indices)


def _compute_scores_by_nature(target_nature, y_pred, y_test, sample_nature, unique_labels, array_of_indices):
    is_target = [x==target_nature for x in sample_nature]
    y_test = list(compress(y_test, is_target))
    y_pred = list(compress(y_pred, is_target))
    return classification_report(y_test, y_pred, labels=unique_labels,
                                                  target_names=array_of_indices, output_dict=True)

