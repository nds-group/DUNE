import ast
import configparser
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts,bm}')



from SPP.SPP import SPP, literal_converter
import re

# ToDo: fix me if needed. This script has become obsolete since #7 was completed.
if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('spp_params.ini')
    use_case = config['DEFAULT']['use_case']
    n_classes = int(config[use_case]['n_classes'])
    n_features = int(config[use_case]['n_features'])
    weights_file = config[use_case]['weights_file']
    f1_file = config[use_case]['f1_file']
    unwanted_classes = ast.literal_eval(config[use_case]['unwanted_classes'])
    experiment_folder_path = config['CORRELATION ANALYSIS']['experiment_folder_path']
    experiment_folder_path_2 = config['CORRELATION ANALYSIS']['experiment_folder_path_2']
    f1_analysis_file_path = config['CORRELATION ANALYSIS']['f1_analysis_file_path']
    start_idx = int(config['CORRELATION ANALYSIS']['start_idx'])
    end_idx = int(config['CORRELATION ANALYSIS']['end_idx'])

    spp = SPP(n_classes=n_classes, n_features=n_features, f1_file=f1_file, unwanted_classes=unwanted_classes, weights_file=weights_file)
    random_experiment_c4_costs = {}
    random_experiment_c4_noF1_costs = {}
    random_experiment_max_theta_costs = {}
    random_experiment_max_theta_7_costs = {}
    random_experiment_max_theta_9_costs = {}
    random_experiment_max_theta_avg_costs = {}
    random_experiment_t7c6_costs = {}
    random_experiment_t8c6_costs = {}
    random_experiment_t9c6_costs = {}
    random_experiment_t4c6_costs = {}

    directory = os.fsencode(experiment_folder_path)
    pattern = re.compile('[0-9]+')

    for file in os.listdir(directory):
        file_string = file.decode("utf-8")
        solution_file_path = os.path.join(experiment_folder_path, file_string)
        if os.path.isdir(solution_file_path) or 'csv' not in file_string:
            # skip directories and must be csv file
            continue
        exp_id = int(pattern.findall(file_string)[1])

        cluster_info = pd.read_csv(f'{experiment_folder_path}/{file.decode("utf-8")}',
                                   converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
        cluster_info = cluster_info.reset_index(drop=True).drop(columns=['Unnamed: 0']).set_index('Cluster')

        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_c4_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_c4_noF1_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_7_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_9_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_avg_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t7c6_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t8c6_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t9c6_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t4c6_costs[exp_id] = sol_cost


    # second directory
    directory = os.fsencode(experiment_folder_path)
    pattern = re.compile('[0-9]+')

    for file in os.listdir(directory):
        file_string = file.decode("utf-8")
        solution_file_path = os.path.join(experiment_folder_path, file_string)
        if os.path.isdir(solution_file_path) or 'csv' not in file_string:
            # skip directories and must be csv file
            continue
        exp_id = 1000+int(pattern.findall(file_string)[1])

        cluster_info = pd.read_csv(f'{experiment_folder_path}/{file.decode("utf-8")}',
                                   converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
        cluster_info = cluster_info.reset_index(drop=True).drop(columns=['Unnamed: 0']).set_index('Cluster')

        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_c4_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_c4_noF1_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_7_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_9_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_max_theta_avg_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t7c6_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t8c6_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t9c6_costs[exp_id] = sol_cost
        sol_cost, sol_groups, sol_feats = spp.encode_cluster_solution(cluster_info.loc[0:])
        random_experiment_t4c6_costs[exp_id] = sol_cost

    random_experiments_df = pd.read_csv(f1_analysis_file_path, delimiter=';')
    random_experiments_df['cost_c4'] = random_experiments_df['exp_id'].apply(lambda x: random_experiment_c4_costs[x])
    random_experiments_df['cost_c4_no_F1'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_c4_noF1_costs[x])
    random_experiments_df['cost_max_Theta_7'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_max_theta_7_costs[x])
    random_experiments_df['cost_max_Theta_9'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_max_theta_9_costs[x])
    random_experiments_df['cost_max_Theta'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_max_theta_costs[x])
    random_experiments_df['cost_max_Theta_Avg'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_max_theta_avg_costs[x])
    random_experiments_df['cost_T7C6'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_t7c6_costs[x])
    random_experiments_df['cost_T8C6'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_t8c6_costs[x])
    random_experiments_df['cost_T9C6'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_t9c6_costs[x])
    random_experiments_df['cost_T4C6'] = random_experiments_df['exp_id'].apply(
        lambda x: random_experiment_t4c6_costs[x])
    random_experiments_df = random_experiments_df.set_index('exp_id', drop=True)
    random_experiments_df = random_experiments_df.assign(n_features_dist=np.nan)
    random_experiments_df = random_experiments_df.astype(dtype={"n_features_dist": "object"})
    random_experiments_df = random_experiments_df.assign(n_classes_dist=np.nan)
    random_experiments_df = random_experiments_df.astype(dtype={"n_classes_dist": "object"})

    for exp_id in list(random_experiments_df.index.values):
        try:
            file_path = f'{experiment_folder_path}/20CL_{exp_id}_SPP_solution.csv'
            cluster_info_df = pd.read_csv(file_path,
                                          converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
        except FileNotFoundError:
            file_path = f'{experiment_folder_path_2}/20CL_{exp_id-1000}_heuristic_solution.csv'
            cluster_info_df = pd.read_csv(file_path,
                                          converters=dict.fromkeys(['Class List', 'Feature List'], literal_converter))
        cluster_info_df['n_features'] = cluster_info_df['Feature List'].str.len()
        cluster_info_df['n_classes'] = cluster_info_df['Class List'].str.len()
        random_experiments_df.at[exp_id, 'n_features_dist'] = cluster_info_df['n_features'].to_list()
        random_experiments_df.at[exp_id, 'n_classes_dist'] = cluster_info_df['n_classes'].to_list()

    random_experiments_df['n_clusters'] = random_experiments_df['n_classes_dist'].str.len()

    corr = random_experiments_df[['F1_macro', 'n_clusters', 'cost_T9C6']].corr(method='spearman')
    with pd.option_context('display.max_columns', None):
        print(corr)
    # palette = sns.color_palette("flare", as_cmap=True)
    palette = sns.cubehelix_palette(rot=-.2, as_cmap=True)

    z_dimension ='F1_macro'
    x_dimension ='n_clusters'
    style='random'
    # style=None

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(data=random_experiments_df, x=x_dimension, y='cost_T7C6', ax=ax, hue=z_dimension, size=z_dimension, style=style, palette=palette, sizes=(10, 100)) # add style='random' to distinguish origin
    ax.set_ylabel('Gain')
    ax.legend(title='n_clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    theta_string = r'$\Theta(\pmb{s_j}) = \max_{\pmb{\phi}} \quad \frac{\pmb{s_j^TW\phi}}{\pmb{1^T\phi}}; \quad \max_{\pmb{x}} \frac{m-\pmb{1^Tx}}{m-1}\cdot\pmb{g^Tx}$'
    ax.set_title(theta_string)
    plt.show()
    #
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(data=random_experiments_df, x=x_dimension, y='cost_T8C6',  ax=ax, hue=z_dimension, size=z_dimension, style=style, palette=palette, sizes=(10, 100))
    ax.set_ylabel('Gain')
    ax.legend(title='n_clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    theta_string = r'$\Theta(\pmb{s_j}) = \max_{\pmb{\phi}} \quad \pmb{s_j^TW\phi} \cdot \frac{r-\pmb{1^T\phi}}{r-1}; \quad \max_{\pmb{x}} \frac{m-\pmb{1^Tx}}{m-1}\cdot\pmb{g^Tx}$'
    ax.set_title(theta_string)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(data=random_experiments_df, x=x_dimension, y='cost_T9C6',  ax=ax, hue=z_dimension, size=z_dimension, style=style, palette=palette, sizes=(10, 100))
    ax.set_ylabel('Gain')
    ax.legend(title='n_clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    theta_string = r'$\Theta(\pmb{s_j}) = \max_{\pmb{\phi}} \quad \frac{\pmb{s_j^TW\phi}}{\pmb{1^T\phi}} \cdot \frac{1}{1+\Delta(\pmb{s_j})}; \quad \max_{\pmb{x}} \frac{m-\pmb{1^Tx}}{m-1}\cdot\pmb{g^Tx}$'
    ax.set_title(theta_string)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(data=random_experiments_df, x=x_dimension, y='cost_c4',  ax=ax, hue=z_dimension, size=z_dimension, style=style, palette=palette, sizes=(10, 100))
    ax.legend(title='n_clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    theta_string = r'$\min_{\pmb{x}} \pmb{c^Tx}; \quad c_j = (m -\Theta(\pmb{s_j})) \cdot \Phi(\pmb{s_j}); \quad \Theta(\pmb{s_j}) = \max_{\pmb{\phi}} \quad \pmb{s_j^TW\phi} - (\tfrac{1}{r}(\pmb{1^Ts_j})\pmb{1})^{\pmb{T}}\pmb{\phi}$'
    ax.set_title(theta_string)
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.scatterplot(data=random_experiments_df, x=x_dimension, y='cost_T4C6', ax=ax, hue=z_dimension, size=z_dimension,
                    style=style, palette=palette, sizes=(10, 100))
    ax.legend(title='n_clusters', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    theta_string = r'$\max_{\pmb{x}} \frac{m-\pmb{1^Tx}}{m-1}\cdot\pmb{g^Tx}; \quad \Theta(\pmb{s_j}) = \max_{\pmb{\phi}} \quad (\pmb{s_j^TW\phi} - (\tfrac{1}{r}(\pmb{1^Ts_j})\pmb{1})^{\pmb{T}}\pmb{\phi}) \cdot \frac{1}{1+\Delta(\pmb{s_j})}$'
    ax.set_title(theta_string)
    ax.set_ylabel('Gain')
    plt.show()


