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

import json
from ast import literal_eval
from collections import defaultdict
from itertools import chain
import os
import re

import warnings

n_trees = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
# Max values per TCAM table, i.e., 85 would require 2 TCAM tables
val_of_max_depth = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

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
                 flow_counts_test_file_path, classes_filter, features_filter, logger):
        self.logger = logger
        self.train_data_folder_path = train_data_folder_path
        self.test_data_folder_path = test_data_folder_path
        self.flow_counts_train_file_path = flow_counts_train_file_path
        self.flow_counts_test_file_path = flow_counts_test_file_path
        self.classes_filter = classes_filter
        self.features_filter = features_filter
        self.feature_list = None
        self.classes = classes_filter
        self.classes_df = pd.DataFrame(classes_filter, columns=['class'])
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

        # # Debug info
        self.logger.debug(f"Train data count: {train_data['Label'].value_counts()}")
        self.logger.debug(f"Test data count: {test_data['Label'].value_counts()}")

        # Assign 'sample_nature' and 'weight' columns
        train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
        test_data['sample_nature'] = test_data.apply(assign_sample_nature, axis=1)

        # Assign 'multiply' column based on the conditions.
        test_data['multiply'] = np.where(test_data['sample_nature']=='pkt', 1, test_data['pkt_count'] - npkts)

        train_data['weight'] = np.where(train_data['sample_nature'] == 'flw',
                                        (train_data['pkt_count'] - npkts + 1) / train_data['pkt_count'],
                                        1 / train_data['pkt_count'])
        return train_data, test_data


    def analyze_models(self, train_data, test_data, filename):

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

        self.write_grid_search(signal_handler, tmp_filename, train_data, test_data)

        shutil.move(tmp_filename, filename)
        self.logger.info(f'Finished model analysis. Saved results to: {filename}')
        signal.signal(signal.SIGINT, original_sigint_handler)
        return []

    def write_grid_search(self, signal_handler, tmp_filename, train_data, test_data):

        # Get Variables and Labels
        y_multiply_test = test_data['multiply']
        flow_pkt_cnt_test = test_data['pkt_count'].to_list()
        flow_ids_test = test_data['Flow ID'].to_list()
        x_train, y_train, samples_nature_train = self.get_x_y_flow(train_data)
        x_test, y_test, samples_nature_test = self.get_x_y_flow(test_data)
        test_label_names, test_indices = self.get_test_labels(test_data)
        print(test_label_names)
        weight_of_samples = list(train_data['weight'])
        self.logger.debug(f'Num Labels: {len(test_label_names)}')

        # ToDo: filter out results with 2 or 4 trees
        with open(tmp_filename, "w") as res_file:
            self.logger.info(f'Writing grid search results to: {tmp_filename}')
            print('depth;tree;no_feats;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;num_samples;'
                  'Macro_F1_PL;Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL',
                  file=res_file)
            # register signal handler to delete file if code is not completed
            signal.signal(signal.SIGINT, signal_handler)
            # FOR EACH (n_tree, leaf, feat)
            for n_tree in n_trees:
                for depth in val_of_max_depth:
                    # get feature orders to use
                    m_feats = get_feature_importance_sets(n_tree, depth, x_train, y_train, weight_of_samples)
                    for feats in m_feats:
                        # ToDo: extract method analyse_grid_point to share code with write_simple_analysis
                        # Prepare a model for the given (depth, n_tree, feat)
                        model = RandomForestClassifier(n_estimators=n_tree, max_depth=depth, n_jobs=10,
                                                       random_state=42, bootstrap=False)
                        # Train (fit) the model with the data
                        model.fit(x_train[feats], y_train, sample_weight=weight_of_samples)
                        # Infer (predict) the labels
                        y_pred = model.predict(x_test[feats]).tolist()

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

                        FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_indices,
                                                                target_names=test_label_names, output_dict=True,
                                                                sample_weight=expanded_weights)

                        macro_f1_FL = FL_class_report['macro avg']['f1-score']
                        weighted_f1_FL = FL_class_report['weighted avg']['f1-score']
                        # If the accuracy and micro avg results are equal, then micro is omitted
                        if 'micro avg' in FL_class_report:
                            micro_f1_FL = FL_class_report['micro avg']['f1-score']
                        else:
                            micro_f1_FL = FL_class_report['accuracy']

                        PL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_indices,
                                                                target_names=test_label_names, output_dict=True)

                        macro_f1_PL = PL_class_report['macro avg']['f1-score']
                        weighted_f1_PL = PL_class_report['weighted avg']['f1-score']
                        # If the accuracy and micro avg results are equal, then micro is omitted
                        if 'micro avg' in PL_class_report:
                            micro_f1_PL = PL_class_report['micro avg']['f1-score']
                        else:
                            micro_f1_PL = PL_class_report['accuracy']

                        # depth = [estimator.tree_.max_depth for estimator in model.estimators_]
                        # The metric on which we select the best model is then the macro_f1_FL
                        # ToDo: evaluate model while searching, and keep in memory the best found so far.
                        #  Save the others as CSV files just in case.
                        print(str(depth) + ';' + str(n_tree) + ';' + str(len(feats)) + ';' +
                              str(macro_f1_FL) + ";" + str(weighted_f1_FL) + ";" + str(micro_f1_FL) + ";" +
                              str(feats) + ';' + str(num_samples) + ';' + str(macro_f1_PL) + ';' + str(weighted_f1_PL) +
                              ';' + str(micro_f1_PL) + ';' + str(FL_class_report) + ';' + str(PL_class_report),
                              file=res_file)

    def analyze_model_n_packets(self, npkts, outfile, force=False):
        print('analyze_model_n_packets')
        """Function for grid search on hyperparameters, features and models"""
        if not force:
            if path.isfile(outfile):
                self.logger.info(
                    f"File {outfile} is present. To overwrite existing files pass force=True when running the"
                    f" analysis")
                return

        train_data, test_data = self.prepare_data(npkts, self.classes_filter)

        self.analyze_models(train_data, test_data, outfile)
        self.logger.debug(f'Analysis completed. Check output file: {outfile}')
    
    def generate_model(self, model_info):
        '''
        Generate a Random Forest model with the given hyper-parameters
        Arguments:
        model_info - A dict including the hyperparameters of the model to generate
        Return:
        model - Random Forest model generated
        FL_class_report - Classification report of the model
        '''
        train_data, test_data = self.prepare_data(model_info['npkts'], self.classes_filter)
        # Get Variables and Labels
        y_multiply_test = test_data['multiply']
        flow_pkt_cnt_test = test_data['pkt_count'].to_list()
        flow_ids_test = test_data['Flow ID'].to_list()
        x_train, y_train, samples_nature_train = self.get_x_y_flow(train_data)
        x_test, y_test, samples_nature_test = self.get_x_y_flow(test_data)
        test_label_names, test_indices = self.get_test_labels(test_data)
        weight_of_samples = list(train_data['weight'])
        model = RandomForestClassifier(n_estimators=model_info['tree'], max_depth=model_info['depth'], n_jobs=10,
                                                       random_state=42, bootstrap=False)
        
        # Train (fit) the model with the data
        model.fit(x_train[model_info['feats']], y_train, sample_weight=weight_of_samples)
        # Infer (predict) the labels
        y_pred = model.predict(x_test[model_info['feats']]).tolist()
        # The result of the flow level inference must be applied to the consecutive packets.
        (expanded_y_test,
            expanded_y_pred,
            expanded_weights,
            expanded_flow_IDs) = extend_test_data_with_flow_level_results(y_test,
                                                                        y_pred,
                                                                        samples_nature_test,
                                                                        y_multiply_test,
                                                                        flow_pkt_cnt_test,
                                                                        flow_ids_test)

        FL_class_report = classification_report(expanded_y_test, expanded_y_pred, labels=test_indices,
                                                target_names=test_label_names, output_dict=True,
                                                sample_weight=expanded_weights)
        
        return model, FL_class_report

    def get_score_per_class(self, cl_report_FL):
        '''
        Get f1 score per class in the model
        Arguments:
        cl_report_FL - Classification report of the selected model (the score in Flow-level)
        Return:
        score_per_class_df - Dataframe including the score and number of samples per class
        '''
        class_names = cl_report_FL.keys()
        score_per_class = []
        classes_ = []
        support_values = []
        for c_name in class_names:
            if c_name in self.classes:
                score_per_class.append(cl_report_FL[c_name]['f1-score'])
                classes_.append(c_name)
                support_values.append(cl_report_FL[c_name]['support'])
                score_per_class_df = pd.DataFrame({'class': classes_, 'f1_score': score_per_class, 'support': support_values})
                score_per_class_df = score_per_class_df.sort_values(by='f1_score', ascending=False)
                score_per_class_df['f1_score'] = score_per_class_df['f1_score']*100
                
        return score_per_class_df
    
    def get_test_labels(self, test_data):
        array_of_indices = []
        unique_labels = test_data["Label"].unique()
        for lab in unique_labels:
            index = self.classes_df[self.classes_df['class'] == lab].index.values[0]
            array_of_indices.append(index)
        return unique_labels, array_of_indices

    def get_x_y_flow(self, dataset):
        # ToDo: selecting the subset of wanted features has nothing to do with the below code.
        x = dataset[self.features_filter]
        # gets the corresponding index for each label. Maybe a scikit requirement?
        y = dataset['Label'].replace(self.classes, range(len(self.classes))).values.tolist()
        sample_nature = dataset['sample_nature']
        return x, y, sample_nature


class UNSWModelAnalyzer(ModelAnalyzer):
    def __init__(self, train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                 flow_counts_test_file_path, classes_filter, features_filter, logger):
        super().__init__(train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                         flow_counts_test_file_path, classes_filter, features_filter, logger)

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
                 flow_counts_test_file_path, classes_filter, features_filter, logger):
        super().__init__(train_data_folder_path, test_data_folder_path, flow_counts_train_file_path,
                         flow_counts_test_file_path, classes_filter, features_filter, logger)

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


def get_feature_importance_sets(n_tree, depth, x_train, y_train, weight_of_samples):
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
    rf_opt = RandomForestClassifier(n_estimators=n_tree, max_depth=depth, random_state=42, bootstrap=False,
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

def select_best_unconstained_model(analysis_files_dir):
    """ Selects the best model based on the macro f1 score
    :param analysis_files_dir: Path to the analysis folder
    :return: the information of selected model as dict
    """
    model_info = {}
    directory = os.fsencode(analysis_files_dir)
    d_frames = []
    pattern = re.compile('[0-9]+')
    
    for file in os.listdir(directory):
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
  
        model_analysis_for_nth = pd.read_csv(f'{analysis_files_dir}/{file_string}', sep=';')
        model_analysis_for_nth['N'] = n_point
        
        d_frames.append(model_analysis_for_nth)
    
    analysis_results_df = pd.concat(d_frames)
    
    #### ORDER in terms of MACRO F1 SCORE and choose the BEST
    chosen_model = analysis_results_df.sort_values('Macro_f1_FL', ascending=0).head(1)
    
    model_info['depth'] = int(chosen_model['depth'].to_list()[0])
    model_info['tree'] = int(chosen_model['tree'].to_list()[0])
    model_info['feats'] = chosen_model['feats'].to_list()[0][2:-2].split('\', \'')
    model_info['npkts'] = chosen_model['N'].to_list()[0]
    model_info['macro_f1_FL'] = chosen_model['Macro_f1_FL'].to_list()[0]
    model_info['weighted_f1_FL'] = chosen_model['Weighted_f1_FL'].to_list()[0]
    model_info['micro_f1_FL'] = chosen_model['Micro_f1_FL'].to_list()[0]
    
    return model_info

