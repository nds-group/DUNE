import ast
import time
import numpy as np
import pandas as pd
import pulp
from matplotlib import pyplot as plt
from tabulate import tabulate
from tqdm import tqdm
import itertools
import logging

from .partitioning import generate_partitions_of_set

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


def get_block_gain(block, importance_matrix, score_vector):
    """Returns the gain of the given block (s_j), provided the per class feature
    importance matrix, W, and the F1 scores vector F. Also provides the set
    of features leading to such cost.

    Parameters
    ----------
    importance_matrix: numpy.ndarray
    The importance matrix
    score_vector: numpy.ndarray
    The F1 score list of the classes.
    block : numpy.ndarray
    The considered grouping

    Returns
    -------
    numpy.float64
        The gain of the block
    numpy.ndarray
        The features selected for the given block
    """
    n_classes = importance_matrix.shape[0]
    n_features = importance_matrix.shape[1]
    feats = (block @ importance_matrix - np.transpose(np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ block))) > 0
    feats = feats.astype(int)
    cost = (block @ (importance_matrix @ feats) - np.ones(n_features) * (1 / n_features) * (np.ones(n_classes) @ block) @ feats)
    f_scores = np.array(list(itertools.compress(score_vector, block)))
    if (sum(block) <= 1):
        f_score_cost = np.max(f_scores)
    else:
        f_score_cost = np.ptp(f_scores)
    cost = cost * (1 /(1 + f_score_cost))
    return cost, feats


def solve_SPP_with_ILP(spp, obj_function, minimize=True, save=False):
    """Finds a solution to the SPP using an Integer Linear Programming (ILP) approach.
    The objective function must be linear.
    Parameters
    ----------
    spp: SPP
    the set partitioning problem object
    obj_function: function
    The objective function used to compute the cost/gain. Make sure it is linear.

    Return
    ----------
    np.ndarray: an array with the cost/gain of each possible block
    list: the list of onehot representations of the features selected for each block
    """

    spp.log.warning('To solve the SPP as an ILP, make sure the cost/gain function is linear.')

    def compute_costs(obj_function):
        """Computes, iteratively, the cost of all blocks, i.e., over 2 ** n_classes -1 groupings
        """
        # we don't want to consider the empty set or the full set
        blocks = list(map(np.array, itertools.product([0, 1], repeat=spp.n_classes)))[1:-1]
        costs = np.zeros(len(blocks))
        features = []
        for block_index, block in enumerate(tqdm(blocks)):
            costs[block_index], feat = obj_function(block, spp.feature_importance, spp.f1_scores)
            features.append(feat)
        return costs, features

    # pre-compute the costs/gain
    cost_values, selected_features = compute_costs(obj_function)

    # cost_values = [obj_function(np.array(onehot(i + 1, spp.n_classes)), spp.feature_importance, spp.f1_scores)[0] for i in range(spp.n_classes)]
    if minimize:
        objective = pulp.LpMinimize
    else:
        objective = pulp.LpMaximize


    possible_partitions = list(map(tuple, itertools.product([0, 1], repeat=spp.n_classes)))[1:-1]
    x = pulp.LpVariable.dicts('part', possible_partitions, lowBound=0, upBound=1, cat=pulp.LpInteger)
    part_model = pulp.LpProblem('Part_Model', objective)
    part_model += pulp.lpSum((cost_values[i] * x[part] for i, part in enumerate(possible_partitions)))

    part_model += (
        pulp.lpSum([x[part] for part in possible_partitions]) <= spp.n_classes - 1,
        "Maximum_number_of_partitions",
    )

    for i in range(spp.n_classes):
        part_model += (
            pulp.lpSum([x[part] for part in possible_partitions if part[i] == 1]) == 1,
            f"Must_include_{onehot(i + 1, spp.n_classes)}",
        )
    part_model.solve(solver=pulp.apis.PULP_CBC_CMD(threads=1, msg=False))

    partition = []
    for part in possible_partitions:
        if x[part].value() == 1.0:
            partition.append(list(part))

    solution_dict = {i: {'classes': [], 'features': []} for i in range(len(partition))}

    for i, part in enumerate(partition):
        part_labels = [c for mask, c in zip(part, spp.classes_list) if mask == 1]
        solution_dict[i]['classes'] = part_labels

    clusters = [sum([2 ** i for i, val in enumerate(reversed(part)) if val == 1]) for part in possible_partitions if
                x[part].value() == 1.0]

    costs = [spp.cost_function[id - 1] for id in clusters]
    cluster_feats = [selected_features[id - 1] for id in clusters]
    accum = 0

    for i, item in enumerate(cluster_feats):
        feats_labels = [f for mask, f in zip(item, spp.features_list) if mask == 1]
        accum = accum + len(feats_labels)
        solution_dict[i]['features'] = feats_labels

    spp.log.info('The ILP Solution:')
    for cluster, cluster_data in solution_dict.items():
        spp.log.info('Cluster %i classes: %s', cluster, cluster_data['classes'])
        spp.log.info('Cluster %i features: %s', cluster, cluster_data['features'])

    spp.log.debug(f'Average number of features: {accum / len(cluster_feats)}')

    if save:
        dst_path = f'{spp.use_case_name}_SPP_solution_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        sol_df = spp.cluster_sol_to_dataframe(partition, cluster_feats)
        sol_df.to_csv(dst_path)
        spp.log.info('Saved solution to %s', dst_path)

    return pulp.value(part_model.objective), solution_dict

class SPP:
    feature_importance = None
    f1_scores = None
    classes_list = None

    def __init__(self, n_classes, n_features, unwanted_classes, use_case, weights_file=None, classes_subset=None,
                 fix_level=None, weights_df=None, f1_file=None, f1_df=None):
        self.fix_level = fix_level
        self.log = logging.getLogger(self.__class__.__name__)
        self.use_case_name = use_case
        self.n_classes = n_classes
        self.n_features = n_features
        if weights_df is None:
            if weights_file is None:
                raise ValueError('Either weights_df or weights_file must be provided')
            self.weights_df = pd.read_csv(weights_file)
        else:
            self.weights_df = weights_df
        self.unwanted_classes = unwanted_classes
        self.classes_subset = classes_subset
        self.features_list = list(self.weights_df.drop(columns=['classes']).columns)
        if f1_df is None:
            if f1_file is None:
                raise ValueError('Either f1_df or f1_file must be provided')
            self.F1_data = pd.read_csv(f1_file).set_index('class')
        else:
            self.F1_data = f1_df


        self.feature_cost = self._get_feature_cost(n_features)
        self.precomputed_gains = None

        self.weights_df = self.weights_df.set_index('classes')
        if classes_subset:
            self.classes_list = self.classes_subset
            self.feature_importance = self.weights_df.loc[self.classes_list].values[:, :self.n_features]
            self.f1_scores = self.F1_data.loc[self.classes_list].sort_values(by='class')[
                'f1_score'].to_list()
        else:
            weights_df = self.weights_df # shortens the below lines
            indices_of_wanted_classes = ~weights_df.index.isin(self.unwanted_classes)
            feature_importance = weights_df.loc[indices_of_wanted_classes].values[:self.n_classes, :self.n_features]
            classes_list = weights_df.loc[indices_of_wanted_classes].index.to_list()[:self.n_classes]

            indices_of_wanted_classes = ~self.F1_data.index.isin(self.unwanted_classes)
            self.F1_data = self.F1_data.loc[indices_of_wanted_classes]
            self.f1_scores = self.F1_data.sort_values(by='class')['f1_score'].to_list()[:self.n_classes]

            self.feature_importance = feature_importance
            self.classes_list = classes_list

    def _get_feature_cost(self, n_features):
        """
        returns the cost of one (any) feature---the cost of features is uniformly distributed
        """
        return (1 / n_features) * (
                np.ones(n_features) * (np.ones(self.n_classes) @ np.array(onehot(2, self.n_classes)))) @ onehot(1,
                                                                                                                n_features)

    def _encode_key(self, key):
        # Initialize a variable to store the total sum
        total_sum = np.zeros((self.n_classes), dtype=int)

        # Iterate through the list and add the values in each tuple to the total sum
        for each_tup in key:
            if type(each_tup) == int:
                total_sum = np.sum([total_sum, onehot(each_tup, self.n_classes)], axis=0)
            else:
                total_sum = np.sum([total_sum, self._encode_key(each_tup)], axis=0)

        return tuple(total_sum)

    def cluster_sol_to_dataframe(self, partition, cluster_feats):
        """
        returns the cluster solution in a DataFrame
        """
        part_labels = [[c for mask, c in zip(part, self.classes_list) if mask == 1] for part in partition]
        feats_labels = [[f for mask, f in zip(item, self.features_list) if mask == 1] for item in cluster_feats]
        cluster_sol = pd.DataFrame({'Cluster': list(range(len(partition))),
                                    'Class List': part_labels,
                                    'Feature List': feats_labels,
                                    })

        self.log.debug(f'Cluster sol to csv: {cluster_sol}')
        return cluster_sol

    def solve_spp_greedy(self, gain_function=get_block_gain, show_plot_gain=True, save=False, print_console=False):
        # it is important that the current_blocks_set are a set, since we will be adding and removing elements
        current_blocks_set = set(range(1, self.n_classes+1)) # the worst case scenario is having a class per cluster

        # Compute the gains for the initial blocks
        gain_map = {}
        blocks_onehot = [onehot(block_id, self.n_classes) for block_id in current_blocks_set]
        for key, key_onehot in zip(current_blocks_set, blocks_onehot):
            gain_map[key] = gain_function(np.array(key_onehot), self.feature_importance, self.f1_scores)

        # Algorithm 1: SPP greedy algorithm
        dendogram = {} # ToDo: rename dendogram dictionary, as it is not a really a dendogram
        gains_list = [] # we cache gains to avoid re-computing them at each level
        for i in range(self.n_classes - 1):
            # generator with all pairs
            all_pairs = [list(tup) for tup in itertools.combinations(current_blocks_set, 2)]
            all_pairs_onehot = [self._encode_key(pair) for pair in all_pairs]

            for key, key_onehot in zip(all_pairs, all_pairs_onehot):
                gain_map[tuple(key)] = gain_function(np.array(key_onehot), self.feature_importance, self.f1_scores)

            best_pair = max(gain_map, key=gain_map.get)

            # Remove invalidated elements from data structures
            to_remove_keys = self._find_keys_of_invalidated_elements(best_pair, current_blocks_set, gain_map)
            for block_index in to_remove_keys:
                gain_map.pop(block_index, None)

            # add to the set of current_blocks_set the new cluster
            current_blocks_set.add(tuple(best_pair))

            # Compute the total gain for the current level
            gain = 0
            features_dict = {}
            for cluster in current_blocks_set:
                gain_delta, cluster_features = gain_function(self._encode_key([cluster]), self.feature_importance, self.f1_scores)
                gain += gain_delta
                features_dict[cluster] = cluster_features
            compactness = ((self.n_classes - len(current_blocks_set))/(self.n_classes - 1))
            total_gain = gain * compactness
            gains_list.append(total_gain)

            # Store the clusters for given level
            dendogram[i] = (current_blocks_set.copy(), features_dict.copy())

        self.precomputed_gains = gains_list
        best_level = np.argmax(np.array(gains_list))

        # If specific level was set, override the best level
        if self.fix_level:
            best_level = self.n_classes - self.fix_level
            best_total_gain = gains_list[best_level]
            self.log.warning(f'The level was fixed at {self.fix_level}, the gain is {best_total_gain}')

        # Process the dendogram to extract the solution
        blocks = []
        features_lists = []
        n_blocks = range(len(dendogram[best_level][0]))
        for block_index in n_blocks:
            block_classes = list(dendogram[best_level][0])[block_index]
            block_features = dendogram[best_level][1][block_classes]
            blocks.append(self._encode_key([block_classes]))
            features_lists.append(block_features)

        dst_path = f'{self.use_case_name}_SPP_solution_LEVEL_{best_level}_{time.strftime("%Y%m%d_%H%M%S")}.csv'
        sol_df = self.cluster_sol_to_dataframe(blocks, features_lists)
        if print_console:
            print(tabulate(sol_df, headers = 'keys', tablefmt = 'psql'))

        if save:
            self.log.info('Saved solution to %s', dst_path)
            sol_df.to_csv(dst_path, index=False)

        if show_plot_gain:
            self.plot_gain(gains_list)
        return sol_df

    def _find_keys_of_invalidated_elements(self, best_pair, current_blocks_set, gain_map):
        to_remove_keys = []
        if type(best_pair) == int: best_pair = [best_pair]
        for elem in best_pair:
            gain_map.pop(elem, None)
            if elem in current_blocks_set: current_blocks_set.remove(elem)
            for key in list(gain_map.keys()):
                if key == best_pair:
                    to_remove_keys.append(key)
                if type(key) == int:
                    if elem == key:
                        to_remove_keys.append(key)
                else:
                    if elem in key:
                        to_remove_keys.append(key)
        return to_remove_keys

    def plot_gain(self, gains_list):
        plt.plot(np.arange(self.n_classes, 1, -1), gains_list)
        plt.xlabel('Level')
        plt.ylabel('Total Gain')
        plt.xlim(self.n_classes, 2)
        plt.xticks(np.arange(self.n_classes, 1, -1))
        plt.show()

    def encode_cluster_solution(self, cluster_info):
        weights = self.weights_df.set_index('classes').sort_values(by='classes')
        weights = weights.loc[~weights.index.isin(self.unwanted_classes)]
        n_classes = len(list(cluster_info['Class List'].explode()))
        n_features = len(list(cluster_info['Feature List'].explode()))
        pcfi_data_no_idx = weights.reset_index()
        pcfi_data_transpose_no_idx = weights.transpose().reset_index()

        try:
            assert n_classes <= self.n_classes
        except AssertionError:
            self.log.error(
                'Number of classes in provided solution must be less or equal than number of classes in the SPP '
                'object')
            exit(1)

        block_gains = []
        classes = []
        features = []
        for cluster_idx, cluster in cluster_info.iterrows():
            cluster_class_list = cluster_info.loc[cluster_idx]["Class List"]
            cluster_features_list = cluster_info.loc[cluster_idx]["Feature List"]
            cluster_W = weights.loc[cluster_class_list][cluster_features_list].values

            # Class list binary encoding
            cluster_class_idx = pcfi_data_no_idx.loc[pcfi_data_no_idx["classes"].isin(cluster_class_list)].index.values
            cluster_class_encoding = np.sum([onehot(idx + 1, n_classes) for idx in cluster_class_idx], axis=0)
            classes.append(cluster_class_encoding)

            f_scores = np.array(list(itertools.compress(self.f1_scores, cluster_class_encoding)))
            if sum(cluster_class_encoding) <= 1:
                psi = 1/(1 + np.max(f_scores))
            else:
                # np.ptp is equivalent to max - min
                psi = 1/(1 + np.ptp(f_scores))

            theta = (np.sum(cluster_W) - self.feature_cost * len(cluster_features_list) * len(cluster_class_list))
            gain = theta * psi

            block_gains.append()

            # Features list binary encoding
            cluster_feats_idx = pcfi_data_transpose_no_idx.loc[
                pcfi_data_transpose_no_idx["index"].isin(cluster_features_list)].index.values
            cluster_feats_encoding = np.sum([onehot(idx + 1, n_features) for idx in cluster_feats_idx], axis=0)
            features.append(cluster_feats_encoding)

        cost = ((n_classes - 1) / (n_classes - cluster_info.shape[0]))
        objective_value = (1/cost) * sum(block_gains)
        return objective_value, classes, features


    def generate_random_spp_solution(self, n_classes=None, n_features=15, costf=get_block_gain):
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

        weights = self.weights_df.set_index('classes').sort_values(by='classes')
        # remove unwanted classes
        weights = weights.loc[~weights.index.isin(self.unwanted_classes)]
        # select the first n_classes and n_features
        weights = weights.sort_values(by='classes').values[:n_classes, :n_features]

        random_partition = get_random_partition(n_classes)
        # select features optimally for the given random partition
        random_feats = [costf(s_j, weights, self.f1_scores)[1] for s_j in random_partition]

        class_names = [[c for mask, c in zip(part, self.classes_list[:n_classes]) if mask == 1] for part in
                       random_partition]

        feature_names = [[c for mask, c in zip(sel, self.features_list) if mask == 1] for sel in random_feats]

        return pd.DataFrame(
            {'Cluster': list(range(len(random_partition))), 'Class List': class_names, 'Feature List': feature_names})


    def solve_spp_brute_force(self, costf=get_block_gain):
        set_elements = list(range(1, self.n_classes + 1))
        all_partitions = generate_partitions_of_set(set_elements)

        all_partitions_onehot = [[np.sum([onehot(idx, self.n_classes) for idx in block], axis=0) for block in partition] for partition in all_partitions]
        all_partitions_gains = []
        for partition in all_partitions_onehot:
            total_gain = 0
            for block in partition:
                gain, feats = costf(block, self.feature_importance, self.f1_scores)
                total_gain += gain
            total_gain = total_gain * ((self.n_classes - len(partition)) / (self.n_classes - 1))
            all_partitions_gains.append(total_gain)

        all_partitions_gains = {idx: gain for idx,gain in enumerate(all_partitions_gains)}
        index = max(all_partitions_gains, key=all_partitions_gains.get)
        gain = all_partitions_gains[index]
        return all_partitions[index], gain
