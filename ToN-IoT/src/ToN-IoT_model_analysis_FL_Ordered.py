# %%
#import useful libraries for analysis and modeling
import pandas as pd
import numpy as np
from sklearn import tree
from scipy import stats
import os
import pickle
import sys
import tempfile
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.tree import export_graphviz, DecisionTreeClassifier
pd.options.mode.chained_assignment = None
from IPython.display import display, HTML
import warnings

# Filter all warnings
warnings.filterwarnings("ignore")


# %%
# list of 8 applications in dataset
# classes = ['Samsung SmartCam', 'Amazon Echo', 'Belkin wemo motion sensor', 'Other']
# classes = ['Light Bulbs LiFX Smart Bulb', 'Smart Things', 'Samsung SmartCam', 'Withings Smart scale', 'Other']
# classes = ['Amazon Echo', 'Laptop', 'Insteon Camera', 'iHome', 'Withings Aura smart sleep sensor']
classes_str = str(pd.read_csv(sys.argv[2])['Class List'].to_list()[int(sys.argv[1])-1])[2:-2]
classes = classes_str.split("', '")
cluster_rank_data = pd.read_csv(sys.argv[3])
other_classes_str =  str(cluster_rank_data[cluster_rank_data['Cluster'] == int(sys.argv[1])-1]['Other_Classes'].to_list()[0])[2:-2]
other_classes = other_classes_str.split("', '")
if other_classes != ['']:
    classes.append('Other')
    
print(sys.argv[1], classes)
print(sys.argv[1], 'OTHER: ', other_classes)
# classes = classes.append('Other')
### Least successful classes
# L_classes = ['MacBook', 'HP Printer', 'PIX-STAR Photo-frame', 'Android Phone', 'TP-Link Smart plug', 'TP-Link Day Night Cloud camera',\
#         'Dropcam', 'Blipcare Blood Pressure meter', 'IPhone']

# classes = ['Dropcam', 'HP Printer', 'Netatmo Welcome', 'Withings Smart Baby Monitor', 'Netatmo weather station',\
#            'Smart Things', 'Amazon Echo', 'Samsung SmartCam','TP-Link Day Night Cloud camera', 'Triby Speaker',\
#               'Belkin Wemo switch', 'TP-Link Smart plug', 'PIX-STAR Photo-frame','Belkin wemo motion sensor',\
#                      'Samsung Galaxy Tab', 'NEST Protect smoke alarm', 'Withings Smart scale', 'IPhone',\
#                             'MacBook', 'Withings Aura smart sleep sensor','Light Bulbs LiFX Smart Bulb',\
#                             'Blipcare Blood Pressure meter','iHome', 'Insteon Camera', 'Android Phone', 'Laptop']
classes_df = pd.DataFrame(classes, columns=['class'])
classes_df = classes_df.reset_index()
# list of all extracted features
feats_all = ["ip.len","ip.ttl","tcp.flags.syn","tcp.flags.ack","tcp.flags.push","tcp.flags.fin","tcp.flags.rst",\
            "tcp.flags.ece","ip.proto","srcport","dstport","ip.hdr_len","tcp.window_size_value","tcp.hdr_len","udp.length",\
            "Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Total","UDP Len Min","UDP Len Max",\
                "Flow IAT Min","Flow IAT Max","Flow IAT Mean","Flow Duration",\
                    "SYN Flag Count","ACK Flag Count","PSH Flag Count","FIN Flag Count","RST Flag Count","ECE Flag Count"]

# list of easy to compute online features - without means
feats_easy = ["ip.len","ip.ttl","tcp.flags.syn","tcp.flags.ack","tcp.flags.push","tcp.flags.fin","tcp.flags.rst",\
            "tcp.flags.ece","ip.proto","srcport","dstport","ip.hdr_len","tcp.window_size_value","tcp.hdr_len","udp.length",\
            "Min Packet Length","Max Packet Length","Packet Length Total","UDP Len Min","UDP Len Max",\
                "Flow IAT Min","Flow IAT Max","Flow Duration","SYN Flag Count","ACK Flag Count",\
                    "PSH Flag Count","FIN Flag Count","RST Flag Count","ECE Flag Count"]

feats_no_time = ["ip.len","ip.ttl","tcp.flags.syn","tcp.flags.ack","tcp.flags.push","tcp.flags.fin","tcp.flags.rst",\
            "tcp.flags.ece","ip.proto","srcport","dstport","tcp.window_size_value","tcp.hdr_len","udp.length",\
            "Min Packet Length","Max Packet Length","Packet Length Total",\
                "SYN Flag Count","ACK Flag Count","PSH Flag Count","FIN Flag Count","RST Flag Count","ECE Flag Count"]

# feats_important = ['dstport', 'tcp.window_size_value', 'srcport', 'ip.ttl', 'ip.len', 'udp.length', 'Packet Length Total', 'Packet Length Mean']
# feats_important = ['ip.len', 'udp.length', 'tcp.hdr_len', 'tcp.window_size_value', 'ip.ttl', 'tcp.flags.rst', 'dstport', 'srcport', 'Packet Length Mean', 'Packet Length Total']
feats_important = pd.read_csv(sys.argv[2])['Feature List'].to_list()[int(sys.argv[1])-1][2:-2].split("', '")
# print(sys.argv[1], feats_important)
# %% [markdown]
#  ### Helper Functions

# %%
""" Function to save trained model to pickle"""
def save_model(RF, filename):
    pickle.dump(RF, open(filename, 'wb'))

def get_test_labels(IoT_Test):
    array_of_indices = []
    unique_labels = IoT_Test["Label_NEW"].unique()
    # unique_labels = IoT_Test["Label"].unique()
    for lab in unique_labels:
        index = classes_df[classes_df['class'] == lab].index.values[0]
        array_of_indices.append(index)
    return unique_labels, array_of_indices

"""
Function to Fit model based on optimal values of depth and number of estimators and use it
to compute feature importance for all the features.
"""
# def get_feature_importance(depth, n_tree, max_leaf, X_train, y_train, weight_of_samples):
    
#     rf_opt = RandomForestClassifier(max_depth = depth, n_estimators = n_tree, max_leaf_nodes=max_leaf, random_state=42, bootstrap=False,n_jobs=10)
#     rf_opt.fit(X_train, y_train, sample_weight=weight_of_samples)
#     feature_importance = pd.DataFrame(rf_opt.feature_importances_)
#     feature_importance.index = X_train.columns
#     feature_importance = feature_importance.sort_values(by=list(feature_importance.columns),axis=0,ascending=False)
    
#     return feature_importance

def get_feature_importance(n_tree, max_leaf, X_train, y_train, weight_of_samples):
    
    rf_opt = RandomForestClassifier(n_estimators = n_tree, max_leaf_nodes=max_leaf, random_state=42, bootstrap=False,n_jobs=10)
    rf_opt.fit(X_train, y_train, sample_weight=weight_of_samples)
    feature_importance = pd.DataFrame(rf_opt.feature_importances_)
    feature_importance.index = X_train.columns
    feature_importance = feature_importance.sort_values(by=list(feature_importance.columns),axis=0,ascending=False)
    
    return feature_importance

"""
Function to Fit model based on optimal values of depth and number of estimators and feature importance
to find the fewest possible features to exceed the previously attained score with all selected features
"""
def get_fewest_features(n_tree, max_leaf, importance):    
    sorted_feature_names = importance.index
    # print('sorted_feature_names: ', sorted_feature_names)
    features = []
    for f in range(1,len(sorted_feature_names)+1):
        features.append(sorted_feature_names[0:f])
    return features

def get_result_scores(classes, cl_report):
    precision=[]
    recall=[]
    f1_score=[]
    supports=[]
    for a_class in classes:
        precision.append(cl_report[a_class]['precision'])
        recall.append(cl_report[a_class]['recall'])
        f1_score.append(cl_report[a_class]['f1-score'])
        supports.append(cl_report[a_class]['support'])
    return precision, recall, f1_score, supports

# def get_scores(classes, depth, n_tree, feats, max_leaf, X_train, y_train, X_test, y_test, unique_labels,array_of_indices,weight_of_samples):
def get_scores(classes, n_tree, feats, max_leaf, X_train, y_train, X_test, y_test, unique_labels,array_of_indices,weight_of_samples):
    model = RandomForestClassifier(n_estimators = n_tree, max_leaf_nodes=max_leaf, n_jobs=10,
                                    random_state=42, bootstrap=False)
    
    model.fit(X_train[feats], y_train, sample_weight=weight_of_samples)
    y_pred = model.predict(X_test[feats])

    # print("##################")
    # print("###PREDICTIONS###", y_pred[0:20])
    # print("###Y_TRUE###", y_test.values[0:20])
    # print("##################")
    # print("###unique_labels###", unique_labels)
    # print("###array_of_indices###", array_of_indices)

    y_test = [int(label) for label in y_test.values]
    y_pred = [int(label) for label in y_pred]

    # print("##################")
    # print("###PREDICTIONS###", y_pred[0:50])
    # print("###Y_TRUE###",      y_test[0:50])

    class_report = classification_report(y_test, y_pred, labels=unique_labels, target_names=array_of_indices, output_dict = True)

    macro_score = class_report['macro avg']['f1-score']
    weighted_score = class_report['weighted avg']['f1-score']

    return model, class_report, macro_score, weighted_score, y_pred

def get_x_y_flow(Dataset, feats):    
    X = Dataset[feats]
    y = Dataset['Label_NEW'].replace(classes, range(len(classes)))
    # y = Dataset['Label'].replace(classes, range(len(classes)))
    sample_nature = Dataset['sample_nature']
    return X, y, sample_nature

def expand_rows_and_get_scores(y_true, y_pred, sample_nature, multiply, test_flow_pkt_cnt, test_flow_IDs, unique_labels, array_of_indices):
    expanded_y_true = []
    expanded_y_pred = []
    #
    expanded_weights = []
    expanded_flow_IDs = []
    
    for true_label, pred_label, nature, mult, pkt_cnt, f_id in zip(y_true, y_pred, sample_nature, multiply, test_flow_pkt_cnt, test_flow_IDs):
        if nature == 'flw':
            expanded_y_true.extend([true_label] * (mult+1))
            expanded_y_pred.extend([pred_label] * (mult+1))
            #
            expanded_weights.extend([1/pkt_cnt] * (mult+1))
            expanded_flow_IDs.extend([f_id]* (mult+1))
        else:
            expanded_y_true.append(true_label)
            expanded_y_pred.append(pred_label)
            #
            expanded_weights.append(1/pkt_cnt)
            expanded_flow_IDs.append(f_id)
    
    # report = classification_report(expanded_y_true, expanded_y_pred)
    
    num_samples = len(expanded_y_true)

    expanded_y_true = [int(label) for label in expanded_y_true]
    expanded_y_pred = [int(label) for label in expanded_y_pred]
    # labels=array_of_indices, target_names=unique_labels,
    cl_report_PL = classification_report(expanded_y_true, expanded_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True)
    macro_f1_PL = cl_report_PL['macro avg']['f1-score']
    weighted_f1_PL = cl_report_PL['weighted avg']['f1-score']
    try:
        micro_f1_PL = cl_report_PL['micro avg']['f1-score']
    except:
        micro_f1_PL = cl_report_PL['accuracy']
    ####
    
    c_report_FL =  classification_report(expanded_y_true, expanded_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True,sample_weight=expanded_weights)
    
    macro_f1_FL = c_report_FL['macro avg']['f1-score']
    weighted_f1_FL = c_report_FL['weighted avg']['f1-score']
    try:
        micro_f1_FL = c_report_FL['micro avg']['f1-score']
    except:
        micro_f1_FL = c_report_FL['accuracy']
    
    return num_samples, macro_f1_PL, weighted_f1_PL, micro_f1_PL, cl_report_PL, macro_f1_FL, weighted_f1_FL, micro_f1_FL, c_report_FL



def compute_flow_pkt_scores(y_pred, y_test, sample_nature,unique_labels,array_of_indices):

    # Create a data frame with the three columns
    df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test, 'sample_nature': sample_nature})
    
    # Split the data frame into two data frames based on sample_nature
    pkt_df = df[df['sample_nature'] == 'pkt']
    flw_df = df[df['sample_nature'] == 'flw']
    
    # Compute macro and weighted F1 scores for pkt_df
    pkt_df_y_true = [int(label) for label in pkt_df['y_test'].values]
    pkt_df_y_pred = [int(label) for label in pkt_df['y_pred']]

    # labels=array_of_indices, target_names=unique_labels,
    pkt_macro_f1 = classification_report(pkt_df_y_true, pkt_df_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True)['macro avg']['f1-score']
    pkt_weighted_f1 = classification_report(pkt_df_y_true, pkt_df_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True)['weighted avg']['f1-score']
    
    # Compute macro and weighted F1 scores for flw_df
    flw_df_y_true = [int(label) for label in flw_df['y_test'].values]
    flw_df_y_pred = [int(label) for label in flw_df['y_pred']]

    flw_macro_f1 = classification_report(flw_df_y_true, flw_df_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True)['macro avg']['f1-score']
    flw_weighted_f1 = classification_report(flw_df_y_true, flw_df_y_pred, labels=unique_labels, target_names=array_of_indices, output_dict=True)['weighted avg']['f1-score']

    return pkt_macro_f1, pkt_weighted_f1, flw_macro_f1, flw_weighted_f1


# %%
# Define a function to check the conditions and assign values
def assign_sample_nature(row):
    if (row["Min Packet Length"] == -1 and
        row["Max Packet Length"] == -1 and
        row["Flow IAT Min"] == -1 and
        row["Flow IAT Max"] == -1):
        return "pkt"
    else:
        return "flw"


# %% [markdown]
#  ### Function for grid search on hyperparameters, features and models

# %%
# Analyze models
def analyze_models(classes, model_type, depths, n_trees, X_train, y_train, X_test, y_test, samples_nature, y_multiply, test_flow_pkt_cnt, test_flow_IDs, max_leaf, test_labels, test_indices, max_feats, filename, weight_of_samples):
    #open file to save ouput of analysis
    with open(filename, "w") as res_file:
        print('depth;tree;no_feats;N_Leaves;Macro_f1_FL;Weighted_f1_FL;Micro_f1_FL;feats;pkt_macro_f1;pkt_weighted_f1;flw_macro_f1;flw_weighted_f1;F1_macro;F1_weighted;num_samples;Macro_F1_PL;Weighted_F1_PL;Micro_F1_PL;cl_report_FL;cl_report_PL', file=res_file)
        if model_type == 'RF':
            # FOR EACH (depth, n_tree, feat)
            for n_tree in n_trees:
                # for depth in depths:
                for leaf in max_leaf:
                    # get feature orders to use
                    # importance = get_feature_importance(depth, n_tree, leaf, X_train, y_train, weight_of_samples)
                    importance = get_feature_importance(n_tree, leaf, X_train, y_train, weight_of_samples)
                    # importance = importance[0:max_feats]
                    m_feats = get_fewest_features(n_tree, leaf, importance) 
                    for feats in m_feats:
                        # feats = feats_important[0:feat_ind]
                        # Get the scores with the given (depth, n_tree, feat)
                        # model, c_report, macro_f1, weight_f1, y_pred = get_scores(classes, depth, n_tree, feats, leaf, X_train, y_train, X_test, y_test,  test_indices, test_labels, weight_of_samples)
                        model, c_report, macro_f1, weight_f1, y_pred = get_scores(classes, n_tree, feats, leaf, X_train, y_train, X_test, y_test,  test_indices, test_labels, weight_of_samples)
                        # 
                        pkt_macro_f1, pkt_weighted_f1, flw_macro_f1, flw_weighted_f1 = compute_flow_pkt_scores(y_pred, y_test, samples_nature, test_indices, test_labels)
                        #
                        num_samples, macro_f1_PL, weighted_f1_PL, micro_f1_PL, cl_report_PL, macro_f1_FL, weighted_f1_FL, micro_f1_FL, cl_report_FL = expand_rows_and_get_scores(y_test, y_pred, samples_nature, y_multiply, test_flow_pkt_cnt, test_flow_IDs, test_indices, test_labels)
                        #
                        # return test_df, macro_f1_FL, weighted_f1_FL, micro_f1_FL, macro_f1_PL, weighted_f1_PL
                        depth = [estimator.tree_.max_depth for estimator in model.estimators_]
                        print(str(depth)+';'+str(n_tree)+';'+str(len(feats))+';'+str(leaf)+";"+str(macro_f1_FL)+";"+str(weighted_f1_FL)+";"+str(micro_f1_FL)+";"+str(list(feats))+';'+str(pkt_macro_f1)+';'+str(pkt_weighted_f1)+';'+str(flw_macro_f1)+';'+str(flw_weighted_f1)+';'+str(macro_f1)+';'+str(weight_f1)+';'+str(num_samples)+';'+str(macro_f1_PL)+';'+str(weighted_f1_PL)+';'+str(micro_f1_PL)+';'+str(cl_report_FL)+';'+str(cl_report_PL), file=res_file)
    print("Analysis Complete. Check output file.")
    return []


# %%
def assign_packet_order(group):
    group['packet'] = range(1, len(group) + 1)
    return group

# %% [markdown]
#  ### Model Analysis - Flows with first n packets

# %%
# Takes desired number of packets and the output file and 
def analyze_model_n_packets(npkts, outfile, feats_to_use, time):    

    # Load Train and Test data
    if(time=="normal"):
        train_data = pd.read_csv("/home/nds-admin/ToN-IoT/data/train/train_"+str(npkts)+"_pkts.csv")
        test_data = pd.read_csv("/home/nds-admin/ToN-IoT/data/test/test_"+str(npkts)+"_pkts.csv")
    #
    flow_pkt_counts = pd.read_csv("/home/nds-admin/ToN-IoT/ToN_IoT_Test_Flow_PktCounts.csv")
    flow_pkt_counts_train = pd.read_csv("/home/nds-admin/ToN-IoT/ToN_IoT_Train_Flow_PktCounts.csv")
    
    flow_count_dict_train = flow_pkt_counts_train.set_index("Flow ID")["packet_counts"].to_dict()
    train_data["pkt_count"] = train_data["Flow ID"].map(flow_count_dict_train)
    #
    # merged_data = test_data.merge(flow_pkt_counts, left_on='Flow ID', right_on='flow.id', how='left')
    # # Extract the count column values and assign them to the pkt_count column in test_data
    # test_data['pkt_count'] = merged_data['count']
    # del merged_data
    flow_count_dict = flow_pkt_counts.set_index("Flow ID")["packet_counts"].to_dict()
    test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)
    all_minus_one = (test_data['Min Packet Length'] == -1) & (test_data['Max Packet Length'] == -1) & (test_data['Packet Length Mean'] == -1)
    # Assign values to the multiply column based on the conditions
    test_data['multiply'] = np.where(all_minus_one, 1, test_data['pkt_count'] - npkts)

    train_data = train_data.sample(frac=1, random_state=42)
    test_data  = test_data.sample(frac=1, random_state=42)

    train_data = train_data.dropna(subset=['srcport', 'dstport']) 
    test_data  = test_data.dropna(subset=['srcport', 'dstport'])
    
    train_data = train_data[(train_data['Label'].isin(classes)) | (train_data['Label'].isin(other_classes))]
    test_data = test_data[(test_data['Label'].isin(classes)) | (test_data['Label'].isin(other_classes))]
    
    # print(train_data['Label'].value_counts())
    # print(test_data['Label'].value_counts())
    ####
    train_data['Label_NEW'] = np.where((train_data['Label'].isin(classes)), train_data['Label'], 'Other')
    test_data['Label_NEW'] = np.where((test_data['Label'].isin(classes)), test_data['Label'], 'Other')
    # print(train_data['Label_NEW'].value_counts())
    # print(test_data['Label_NEW'].value_counts())
    
    ####

    test_labels, test_indices = get_test_labels(test_data)
    print("Num Labels: ", len(test_labels))
    # print("Index: ", test_indices)

    train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
    test_data['sample_nature']  = test_data.apply(assign_sample_nature, axis=1)
    
    # train_data['weight'] = np.where(train_data['sample_nature'] == 'flw', npkts, 1)
    # weight_of_samples = list(train_data['weight'])
    
    train_data['weight'] = np.where(train_data['sample_nature'] == 'flw', (train_data['pkt_count'] - npkts + 1)/train_data['pkt_count'], 1/train_data['pkt_count'])
    weight_of_samples = list(train_data['weight'])

    # Get Variables and Labels
    y_multiply = test_data['multiply'].astype(int)
    test_flow_pkt_cnt = test_data['pkt_count'].to_list()
    test_flow_IDs = test_data['Flow ID'].to_list()
    X_train, y_train, sample_nat_train = get_x_y_flow(train_data, feats_to_use)
    X_test,  y_test, sample_nat_test  = get_x_y_flow(test_data, feats_to_use)

    # leaves   = [500]
    # depths   = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    depths = []
    trees    = [1,2,3,4,5]
    val_of_max_leaves = [41, 85, 129, 173, 217, 261, 305, 349, 393, 437, 481, 500]
    # trees = [5]
    max_feat = len(feats_important)

    analyze_models(classes, "RF", depths, trees, X_train, y_train, X_test, y_test, sample_nat_test, y_multiply, test_flow_pkt_cnt,test_flow_IDs, val_of_max_leaves, test_labels, test_indices, max_feat, outfile, weight_of_samples)

# # #### 1st n packets
for nd in range(3,5):
    print("Number of Packets for Flow Features: ", nd)
    f_name = "/home/nds-admin/distributed_in_band/ToN-IoT/model_analysis_results/clustering_results/6CLuster_Ordered/ToN-IoT_models_"+str(nd)+"pkts_10CL_Cluster"+str(int(sys.argv[1])-1)+"_MACRO_Ordered_withOptimizer_NOdelta_051342.csv"
    analyze_model_n_packets(nd, f_name, feats_important, "normal")



# %%



