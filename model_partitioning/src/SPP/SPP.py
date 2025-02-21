import ast
import time
import numpy as np
import pandas as pd
import pulp
from matplotlib import pyplot as plt
from operator import itemgetter
from tabulate import tabulate
from tqdm import tqdm
import itertools
import operator
import logging

import os

from SPP.partitioning import generate_partitions_of_set

logging.basicConfig(level=logging.INFO)


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    return val if val == '' else ast.literal_eval(val)


def onehot(y, max):
    """Returns the one-hot encoding of the input

    Parameters
    ----------
    y: int
    The value to be encoded
    max: int
    The max encodable value, i.e, the length of the array.

    Returns
    -------
    tuple
        The one-hot encoding of the input, y.
    """
    return tuple([1 if x + 1 == y else 0 for x in range(max)])


def positions_of_ones(binary_array):
    """
    Given a binary array, returns a tuple with the positions where the binary array contains 1s.

    Parameters:
    binary_array (list of int): A list of integers (0s and 1s).

    Returns:
    tuple: A tuple containing the indices of the elements that are 1.
    """
    return tuple(index for index, value in enumerate(binary_array) if value == 1)


def compute_group_cost_maximize(s_j, W, F=None):
    """Returns the cost of grouping s_j, provided the per class feature
    importance matrix, W, and the F1 scores list F. Also provides the set
    of features leading to such cost.

    Parameters
    ----------
    W: numpy.ndarray
    The importance matrix
    F: numpy.ndarray
    The F1 score list of the classes.
    s_j : numpy.ndarray
    The considered grouping

    Returns
    -------
    numpy.float64
        The cost of the grouping
    numpy.ndarray
        The features selected for the grouping
    """
    n_classes = W.shape[0]
    n_features = W.shape[1]
    feats = (s_j @ W - np.transpose(np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j))) > 0
    feats = feats.astype(int)
    cost = (s_j @ (W @ feats) - np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j) @ feats)
    return cost, feats


def compute_group_cost_c4(s_j, W, F):
    """Returns the cost of grouping s_j, provided the per class feature
    importance matrix, W, and the F1 scores list F. Also provides the set
    of features leading to such cost.

    Parameters
    ----------
    W: numpy.ndarray
    The importance matrix
    F: numpy.ndarray
    The F1 score list of the classes.
    s_j : numpy.ndarray
    The considered grouping

    Returns
    -------
    numpy.float64
        The cost of the grouping
    numpy.ndarray
        The features selected for the grouping
    """
    n_classes = W.shape[0]
    n_features = W.shape[1]
    feats = (s_j @ W - np.transpose(np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j))) > 0
    feats = feats.astype(int)
    cost = n_classes - (s_j @ (W @ feats) - np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j) @ feats)
    f_scores = np.array(list(itertools.compress(F, s_j)))
    if (sum(s_j) <= 1):
        f_score_cost = np.max(f_scores)
    else:
        f_score_cost = np.ptp(f_scores)
    cost = cost * f_score_cost
    return cost, feats


def compute_group_gain_c4(s_j, W, F):
    """Returns the cost of grouping s_j, provided the per class feature
    importance matrix, W, and the F1 scores list F. Also provides the set
    of features leading to such cost.

    Parameters
    ----------
    W: numpy.ndarray
    The importance matrix
    F: numpy.ndarray
    The F1 score list of the classes.
    s_j : numpy.ndarray
    The considered grouping

    Returns
    -------
    numpy.float64
        The cost of the grouping
    numpy.ndarray
        The features selected for the grouping
    """
    n_classes = W.shape[0]
    n_features = W.shape[1]
    feats = (s_j @ W - np.transpose(np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j))) > 0
    feats = feats.astype(int)
    cost = (s_j @ (W @ feats) - np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j) @ feats)
    f_scores = np.array(list(itertools.compress(F, s_j)))
    if (sum(s_j) <= 1):
        f_score_cost = np.max(f_scores)
    else:
        f_score_cost = np.ptp(f_scores)
    cost = cost * (1 /(1 + f_score_cost))
    return cost, feats


def compute_group_cost_c4_no_F1(s_j, W, F=None):
    """Returns the cost of grouping s_j, provided the per class feature
    importance matrix, W, and the F1 scores list F. Also provides the set
    of features leading to such cost.

    Parameters
    ----------
    W: numpy.ndarray
    The importance matrix
    F: numpy.ndarray
    The F1 score list of the classes.
    s_j : numpy.ndarray
    The considered grouping

    Returns
    -------
    numpy.float64
        The cost of the grouping
    numpy.ndarray
        The features selected for the grouping
    """
    n_classes = W.shape[0]
    n_features = W.shape[1]
    feats = (s_j @ W - np.transpose(np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j))) > 0
    feats = feats.astype(int)
    cost = n_classes - (s_j @ (W @ feats) - np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ s_j) @ feats)
    return cost, feats


def compute_group_gain(s_j, W, F=None):
    """Returns the gain of grouping s_j, provided the per class feature
    importance matrix, W, and the F1 scores list F. Also provides the set
    of features leading to such cost.

    Parameters
    ----------
    W: numpy.ndarray
    The importance matrix
    F: numpy.ndarray
    The F1 score list of the classes.
    s_j : numpy.ndarray
    The considered grouping

    Returns
    -------
    numpy.float64
        The gain of the grouping
    numpy.ndarray
        The features selected for the grouping
    """
    n_classes = W.shape[0]
    n_features = W.shape[1]
    phis = list(map(np.array, itertools.product([0, 1], repeat=n_features)))[1:-1]
    max_values = np.array(
        [s_j @ (W @ phi) / (np.ones(n_features) @ phi) for phi in phis])
    res = np.argmax(max_values)
    theta = max_values[res]
    f_scores = np.array(list(itertools.compress(F, s_j)))
    if (sum(s_j) <= 1):
        f_score_gain = np.max(f_scores)
    else:
        f_score_gain = np.ptp(f_scores)
    gain = theta * (1 / (1 + f_score_gain))
    return gain, phis[res]


class SPP:
    W = None
    F = None
    classes_list = None

    def __init__(self, n_classes, n_features, unwanted_classes, name, weights_file=None, test_classes_list=None,
                 fix_level=None, weights_df=None, f1_file=None, f1_df=None):
        self.fix_level = fix_level
        self.log = logging.getLogger(self.__class__.__name__)
        self.use_case_name = name
        self.n_classes = n_classes
        self.n_features = n_features
        if weights_df is None:
            if weights_file is None:
                raise ValueError('Either weights_df or weights_file must be provided')
            self.weights_df = pd.read_csv(weights_file)
        else:
            self.weights_df = weights_df
        self.unwanted_classes = unwanted_classes
        self.test_classes_list = test_classes_list
        self.features_list = list(self.weights_df.drop(columns=['c_name']).columns)
        if f1_df is None:
            if f1_file is None:
                raise ValueError('Either f1_df or f1_file must be provided')
            self.F1_data = pd.read_csv(f1_file).set_index('class').drop(columns=['Unnamed: 0'])
        else:
            self.F1_data = f1_df
        self.feature_cost = (1 / n_features) * (
                np.ones(n_features) * (np.ones(self.n_classes) @ np.array(onehot(2, self.n_classes)))) @ onehot(1,
                                                                                                                n_features)
        self.gains_list = None
        self.c = None
        if test_classes_list:
            self.generate_test_problem_data()
        else:
            self.generate_problem_data()

    def generate_test_problem_data(self):
        self.classes_list = self.test_classes_list
        W = self.weights_df.set_index('c_name').loc[self.classes_list].values[:, :self.n_features]
        F = self.F1_data.loc[self.classes_list].sort_values(by='class')[
            'f1_score'].to_list()
        self.W = W
        self.F = F

    def generate_problem_data(self):
        pcfi_data = self.weights_df.set_index('c_name')
        W = pcfi_data.loc[~pcfi_data.index.isin(self.unwanted_classes)].values[:self.n_classes, :self.n_features]
        classes_list = pcfi_data.loc[~pcfi_data.index.isin(self.unwanted_classes)].index.to_list()[:self.n_classes]

        self.F1_data = self.F1_data.loc[~self.F1_data.index.isin(self.unwanted_classes)]
        F = self.F1_data.sort_values(by='class')['f1_score'].to_list()[:self.n_classes]

        self.W = W
        self.F = F
        self.classes_list = classes_list

    def compute_costs(self, costf=compute_group_cost_c4):
        """Computes, iteratively, the cost of all blocks, i.e., over 2 ** n_classes -1 groupings
        """
        S = list(map(np.array, itertools.product([0, 1], repeat=self.n_classes)))[1:-1]
        c = np.zeros(len(S))
        feats = []
        for j, s_j in enumerate(tqdm(S)):
            c[j], feat = costf(s_j, self.W, self.F)
            feats.append(feat)
        self.c = c
        self.feats = feats

    def solve_SPP_with_ILP(self, minimize=True, save=False):
        if self.c is None:
            self.log.warning('Cost is None. Computing using the default cost function...')
            self.compute_costs()
        if minimize:
            objective = pulp.LpMinimize
        else:
            objective = pulp.LpMaximize

        possible_partitions = list(map(tuple, itertools.product([0, 1], repeat=self.n_classes)))[1:-1]
        x = pulp.LpVariable.dicts('part', possible_partitions, lowBound=0, upBound=1, cat=pulp.LpInteger)
        part_model = pulp.LpProblem('Part_Model', objective)
        part_model += pulp.lpSum((self.c[i] * x[part] for i, part in enumerate(possible_partitions)))

        part_model += (
            pulp.lpSum([x[part] for part in possible_partitions]) <= self.n_classes - 1,
            "Maximum_number_of_partitions",
        )

        for i in range(self.n_classes):
            part_model += (
                pulp.lpSum([x[part] for part in possible_partitions if part[i] == 1]) == 1,
                f"Must_include_{onehot(i + 1, self.n_classes)}",
            )
        part_model.solve(solver=pulp.apis.PULP_CBC_CMD(threads=1, msg=0))

        partition = []
        for part in possible_partitions:
            if x[part].value() == 1.0:
                partition.append(list(part))

        solution_dict = {i: {'classes': [], 'features': []} for i in range(len(partition))}

        for i, part in enumerate(partition):
            part_labels = [c for mask, c in zip(part, self.classes_list) if mask == 1]
            solution_dict[i]['classes'] = part_labels
            # self.log.info(part_labels)
            # self.log.debug("ENCODING:", part)
            # self.log.info()

        clusters = [sum([2 ** i for i, val in enumerate(reversed(part)) if val == 1]) for part in possible_partitions if
                    x[part].value() == 1.0]

        costs = [self.c[id - 1] for id in clusters]
        cluster_feats = [self.feats[id - 1] for id in clusters]
        accum = 0

        for i, item in enumerate(cluster_feats):
            feats_labels = [f for mask, f in zip(item, self.features_list) if mask == 1]
            accum = accum + len(feats_labels)
            solution_dict[i]['features'] = feats_labels
            # self.log.info(feats_labels)
            # self.log.debug("ENCODING:", item)

        self.log.info('The ILP Solution:')
        for cluster, cluster_data in solution_dict.items():
            self.log.info('Cluster %i classes: %s', cluster, cluster_data['classes'])
            self.log.info('Cluster %i features: %s', cluster, cluster_data['features'])

        self.log.debug(f'Average number of features: {accum / len(cluster_feats)}')

        if save:
            dst_path = f'{self.use_case_name}_SPP_solution_{time.strftime("%Y%m%d_%H%M%S")}.csv'
            sol_df = self.cluster_sol_to_csv(partition, cluster_feats)
            sol_df.to_csv(dst_path)
            self.log.info('Saved solution to %s', dst_path)

        return pulp.value(part_model.objective), solution_dict

    def cluster_sol_to_csv(self, partition, cluster_feats):
        # for part, cluster_feats in zip(partition, cluster_feats):
        part_labels = [[c for mask, c in zip(part, self.classes_list) if mask == 1] for part in partition]
        feats_labels = [[f for mask, f in zip(item, self.features_list) if mask == 1] for item in cluster_feats]
        cluster_sol = pd.DataFrame({'Cluster': list(range(len(partition))),
                                    'Class List': part_labels,
                                    'Feature List': feats_labels,
                                    # 'Depth': [-1] * len(partition),
                                    # 'Tree': [-1] * len(partition),
                                    # 'Feats': [-1] * len(partition)
                                    })

        self.log.debug(f'Cluster sol to csv: {cluster_sol}')
        return cluster_sol

    def solve_SPP_with_heuristic(self, gain_f=compute_group_gain_c4, plot_gain=True, save=False, print_console=False):
        distances = np.zeros(self.n_classes-1)
        clusters_list = set(range(1, self.n_classes + 1))
        clusters_onehot_list = [onehot(x, self.n_classes) for x in clusters_list]

        gain_dict = {key: gain_f(np.array(key_onehot), self.W, self.F) for key, key_onehot in
                     zip(clusters_list, clusters_onehot_list)}
        dendogram = {}

        # gain_data_store = gain_dict.copy()

        def encode_key(key):
            # Initialize a variable to store the total sum
            total_sum = np.zeros((self.n_classes), dtype=int)

            # Iterate through the list and add the values in each tuple to the total sum
            for each_tup in key:
                if type(each_tup) == int:
                    total_sum = np.sum([total_sum, onehot(each_tup, self.n_classes)], axis=0)
                else:
                    total_sum = np.sum([total_sum, encode_key(each_tup)], axis=0)

            return tuple(total_sum)

        gains_list = []

        for i in range(self.n_classes - 1):
            # print('-' * width)
            # print('')
            # print(f'Level {self.n_classes - i}')
            # print('')
            # print('-' * width)
            all_pairs = [list(tup) for tup in itertools.combinations(clusters_list, 2)]  # generator with all pairs
            all_pairs_onehot = [encode_key(x) for x in all_pairs]

            for key, key_onehot in zip(all_pairs, all_pairs_onehot):
                # if tuple(key) not in gain_dict:
                gain_dict[tuple(key)] = gain_f(np.array(key_onehot), self.W, self.F)
                # gain_data_store[tuple(key)] = gain_dict[tuple(key)]

            best_pair = max(gain_dict.items(), key=operator.itemgetter(1))[0]
            distance = max(gain_dict.items(), key=operator.itemgetter(1))[1][0]

            # Remove invalidated elements from data structures
            to_remove_keys = []
            if type(best_pair) == int: best_pair = [best_pair]
            for elem in best_pair:
                gain_dict.pop(elem, None)
                if elem in clusters_list: clusters_list.remove(elem)
                for key in list(gain_dict.keys()):
                    if key == best_pair:
                        to_remove_keys.append(key)
                    if type(key) == int:
                        if elem == key:
                            to_remove_keys.append(key)
                    else:
                        if elem in key:
                            to_remove_keys.append(key)
            for k in to_remove_keys:
                gain_dict.pop(k, None)

            # add to the cluster list
            distances[i] = distance
            clusters_list.add(tuple(best_pair))

            # Compute the total gain for the current level
            gain = 0
            features_dict = {}
            for cluster in clusters_list:
                gain_delta, cluster_features = gain_f(encode_key([cluster]), self.W, self.F)
                gain += gain_delta
                features_dict[cluster] = cluster_features
            compactness = ((self.n_classes - len(clusters_list))/(self.n_classes - 1))
            total_gain = gain * compactness
            gains_list.append(total_gain)

            # Store the clusters for given level
            dendogram[i] = (clusters_list.copy(), features_dict.copy())

            # print(f'Best pair: {best_pair}')
            # print(f'Clusters: {clusters_list}')
            # print(f'Gain dict: {gain_dict}    ')
            # print('')

        best_level = np.argmax(np.array(gains_list))
        best_total_gain = gains_list[best_level]
        # print(f'Best level: {self.n_classes - best_level}')
        # print(f'Best gain: {best_total_gain}')

        # If specific level was set, override the best level
        if self.fix_level:
            best_level = self.n_classes - self.fix_level
            best_total_gain = gains_list[best_level]
            # print(f'Warning: level was fixed at {self.fix_level}, the gain is {best_total_gain}')
        # print(f'The clusters are:')
        partitions_list = []
        features_list = []
        clusters_data = ((k, list(dendogram[best_level][0])[k], dendogram[best_level][1][list(dendogram[best_level][0])[k]]) for k in range(len(dendogram[best_level][0])))
        for item in clusters_data:
            j, cluster_classes, cluster_feats = item
            partitions_list.append(encode_key([cluster_classes]))
            features_list.append(cluster_feats)
            # print(f'\t[{j}]:{cluster_classes} \t features mask:{cluster_feats}')

        dst_path = f'{self.use_case_name}_SPP_solution_LEVEL_{best_level}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        sol_df = self.cluster_sol_to_csv(partitions_list, features_list)
        if print_console:
            print(tabulate(sol_df, headers = 'keys', tablefmt = 'psql'))
            # print(f'Distances array: {distances}')
            # plt.plot(np.arange(self.n_classes - 1), distances)
            # plt.show()
        if save:
            self.log.info('Saved solution to %s', dst_path)
            sol_df.to_csv(dst_path)

        if plot_gain:
            plt.plot(np.arange(self.n_classes, 1, -1), gains_list)
            plt.xlabel('Level')
            plt.ylabel('Total Gain')
            plt.xlim(self.n_classes, 2)
            plt.xticks(np.arange(self.n_classes, 1, -1))
            plt.show()
        self.gains_list = gains_list
        return sol_df

    def solve_mock_SPP_with_heuristic(self):
        n = 5
        clusters_list = list(range(1, n + 1))
        gain_dict = {key: 0 for key in clusters_list}
        def compute_nested_sum(tuple_list):
            # Initialize a variable to store the total sum
            total_sum = 0

            # Iterate through the list and add the values in each tuple to the total sum
            for each_tup in tuple_list:
                if type(each_tup) == int:
                    total_sum += each_tup
                else:
                    total_sum += compute_nested_sum(each_tup)

            return total_sum

        for i in range(n-1):
            # print('-' * width)
            # print(f'Level {n-i}')
            # print('-' * width)
            all_pairs = (list(tup) for tup in itertools.combinations(clusters_list, 2)) #  generator with all pairs
            for key in all_pairs:
                gain_dict[tuple(key)] = compute_nested_sum(key)


            best_pair = max(gain_dict.items(), key=operator.itemgetter(1))[0]
            clusters_list.append(best_pair)
            to_remove_keys = []
            for elem in best_pair:
                gain_dict.pop(elem)
                if elem in clusters_list: clusters_list.remove(elem)
                for key in list(gain_dict.keys()):
                    if key == best_pair:
                        continue
                    if type(key) == int:
                        if elem == key:
                            to_remove_keys.append(key)
                    else:
                        if elem in key:
                            to_remove_keys.append(key)

            for k in to_remove_keys:
                gain_dict.pop(k, None)

            print(f'Best pair: {best_pair}')
            print(f'Clusters: {clusters_list}')
            print(f'Gain dict: {gain_dict}')
            print('')



    def encode_cluster_solution(self, cluster_info, costf='C4'):
        W_df = self.weights_df.set_index('c_name').sort_values(by='c_name')
        W_df = W_df.loc[~W_df.index.isin(self.unwanted_classes)]
        n_classes = len(list(cluster_info['Class List'].explode()))
        n_features = len(list(cluster_info['Feature List'].explode()))
        pcfi_data_no_idx = W_df.reset_index()
        pcfi_data_transpose_no_idx = W_df.transpose().reset_index()

        try:
            assert n_classes <= self.n_classes
        except AssertionError:
            self.log.error(
                'Number of classes in provided solution must be less or equal than number of classes in the SPP '
                'object')
            exit(1)

        groups_cost = []
        class_lists = []
        feats_lists = []
        # ToDo: refactor and extract the cost functions
        for cluster_idx, cluster in cluster_info.iterrows():
            cluster_class_list = cluster_info.loc[cluster_idx]["Class List"]
            cluster_features_list = cluster_info.loc[cluster_idx]["Feature List"]
            cluster_W = W_df.loc[cluster_class_list][cluster_features_list].values

            # Class list binary encoding
            cluster_class_idx = pcfi_data_no_idx.loc[pcfi_data_no_idx["c_name"].isin(cluster_class_list)].index.values
            cluster_class_encoding = np.sum([onehot(idx + 1, n_classes) for idx in cluster_class_idx], axis=0)
            class_lists.append(cluster_class_encoding)

            f_scores = np.array(list(itertools.compress(self.F, cluster_class_encoding)))
            if (sum(cluster_class_encoding) <= 1):
                f_score_cost = np.max(f_scores)
            else:
                f_score_cost = np.ptp(f_scores)

            # first compute Theta for C4.
            if 'C4' in costf:
                # this one is for minimization
                theta = (np.sum(cluster_W) - self.feature_cost * len(cluster_features_list) * len(cluster_class_list))
                cost = n_classes - theta
                # Then if the F1 score is not excluded, e.g., costf != 'C4 (no F1)'. In other words costf == C4
                if 'C4' == costf:
                    # Consider the PTP F1 score in the cost
                    cost = cost * f_score_cost
            elif ('max Theta 7' == costf):
                theta = np.sum(cluster_W) / len(cluster_features_list)
                cost = theta
            elif ('max Theta 9' == costf):
                theta = np.sum(cluster_W) / len(cluster_features_list)
                theta = theta * (1 / (1 + f_score_cost))
                cost = theta
            elif 'max Theta' == costf:
                theta = (np.sum(cluster_W) - self.feature_cost * len(cluster_features_list) * len(cluster_class_list))
                cost = theta
            elif 'max Theta Avg.' == costf:
                theta = (np.sum(cluster_W) - self.feature_cost * len(cluster_features_list) * len(cluster_class_list))
                cost = theta / cluster_info.shape[0]
            elif 'T7C6' == costf:
                theta = np.sum(cluster_W) / len(cluster_features_list)
                # cost = theta * ((n_classes - cluster_info.shape[0]) / (n_classes - 1))
                cost = theta
            elif 'T8C6' == costf:
                theta = np.sum(cluster_W) * ((self.n_features - len(cluster_features_list)) / (self.n_features - 1))
                # cost = theta * ((n_classes - cluster_info.shape[0]) / (n_classes - 1))
                cost = theta
            elif 'T9C6' == costf:
                theta = np.sum(cluster_W) * ((self.n_features - len(cluster_features_list)) / (self.n_features - 1))
                theta = theta * (1 / (1 + f_score_cost))
                cost = theta
                # cost = theta * ((n_classes - cluster_info.shape[0]) / (n_classes - 1))
            elif 'T4C6' == costf:
                theta = (np.sum(cluster_W) - self.feature_cost * len(cluster_features_list) * len(cluster_class_list))
                theta = theta * (1 / (1 + f_score_cost))
                cost = theta
                # cost = theta * ((n_classes - cluster_info.shape[0]) / (n_classes - 1))
            else:
                raise ValueError("Unexpected value for parameter costf. Valid values are: {'C4', 'C4 (no F1), 'C6'}")

            groups_cost.append(cost)

            # Features list binary encoding
            cluster_feats_idx = pcfi_data_transpose_no_idx.loc[
                pcfi_data_transpose_no_idx["index"].isin(cluster_features_list)].index.values
            cluster_feats_encoding = np.sum([onehot(idx + 1, n_features) for idx in cluster_feats_idx], axis=0)
            feats_lists.append(cluster_feats_encoding)

        total_cost = sum(groups_cost)
        if 'C6' in costf:
            total_cost = total_cost * ((n_classes - cluster_info.shape[0]) / (n_classes - 1))
        return total_cost, class_lists, feats_lists


    def generate_random_spp_solution(self, n_classes=None, n_features=15, costf=compute_group_gain_c4):
        """Generate a random solution for the SPP problem
        Parameters
        ----------
        n_classes: int or None
        overwrites the number of classes that is considered for the set partitioning problem
        n_features: int
        the number of features considered for the set partitioning problem
        costf: function
        The cost function used to compute the cost/gain. See compute_group_gain_c4() for an example on the structure.

        Return
        ----------
        pd.DataFrame: the partition representation in a DataFrame
        """
        if n_classes is None:
            n_classes = self.n_classes
        try:
            assert n_classes <= self.n_classes
        except AssertionError:
            self.log.error('Number of classes in random solution must not be greater than number of classes in the SPP object')
            exit(1)

        # this is a lazy import, since this function is not always used
        from random import shuffle, sample, randint

        def encode_int_encoded_partition(partition, n_classes):
            return np.sum([onehot(x, n_classes) for x in partition], axis=0)

        def get_random_partition(n_classes):
            """
            Returns a np.ndarray representing a random partition of the SPP---a valid solution.
            """
            indices = list(np.arange(1, n_classes + 1))

            # select number_of_blocks at random
            number_of_blocks = randint(2, n_classes - 1)

            # Shuffle the list of all possible blocks
            shuffle(indices)

            # Initialize the output lists
            partition = []

            # Generate random lengths for each block in the partition
            lengths = sorted(sample(range(0, n_classes), number_of_blocks - 1))
            lengths.append(n_classes)

            # Use the lengths to slice the indices list and create the partition
            start = 0
            for end in lengths:
                partition.append(indices[start:end])
                start = end

            return np.stack(
                [encode_int_encoded_partition(partition, n_classes) for partition in partition if len(partition) > 0],
                axis=0)

        weights = self.weights_df.set_index('c_name').sort_values(by='c_name')
        # remove unwanted classes
        weights = weights.loc[~weights.index.isin(self.unwanted_classes)]
        # select the first n_classes and n_features
        weights = weights.sort_values(by='c_name').values[:n_classes, :n_features]

        random_partition = get_random_partition(n_classes)
        # select features optimally for the given random partition
        random_feats = [costf(s_j, weights, self.F)[1] for s_j in random_partition]

        class_names = [[c for mask, c in zip(part, self.classes_list[:n_classes]) if mask == 1] for part in
                       random_partition]

        feature_names = [[c for mask, c in zip(sel, self.features_list) if mask == 1] for sel in random_feats]

        return pd.DataFrame(
            {'Cluster': list(range(len(random_partition))), 'Class List': class_names, 'Feature List': feature_names})


    def solve_SPP_brute_force(self, costf=compute_group_gain_c4):
        set_elements = list(range(1, self.n_classes + 1))
        all_partitions = generate_partitions_of_set(set_elements)

        all_partitions_onehot = [[np.sum([onehot(idx, self.n_classes) for idx in block], axis=0) for block in partition] for partition in all_partitions]
        all_partitions_gains = []
        for partition in all_partitions_onehot:
            total_gain = 0
            for block in partition:
                gain, feats = costf(block, self.W, self.F)
                total_gain += gain
            total_gain = total_gain * ((self.n_classes - len(partition)) / (self.n_classes - 1))
            all_partitions_gains.append(total_gain)
        index, gain = max(enumerate(all_partitions_gains), key=itemgetter(1))

        return all_partitions[index], gain
