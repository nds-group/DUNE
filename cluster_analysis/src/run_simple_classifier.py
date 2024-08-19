from random import sample

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from shap import TreeExplainer, summary_plot

val_of_max_leaves = [41, 85, 129, 173, 217, 261, 305, 349, 393, 437, 481, 500]
max_leaf = val_of_max_leaves[-1]

classes = ['Amazon Echo', 'Android Phone', 'Belkin Wemo switch', 'Belkin wemo motion sensor', 'Dropcam', 'HP Printer',
           'Insteon Camera', 'Laptop', 'Light Bulbs LiFX Smart Bulb', 'MacBook', 'NEST Protect smoke alarm',
           'Netatmo Welcome', 'Netatmo weather station', 'PIX-STAR Photo-frame', 'Samsung Galaxy Tab',
           'Samsung SmartCam', 'Smart Things', 'TP-Link Day Night Cloud camera', 'TP-Link Smart plug',
           'Triby Speaker', 'Withings Aura smart sleep sensor', 'Withings Smart Baby Monitor',
           'Withings Smart scale', 'iHome']
features = ['Flow Duration', 'Flow IAT Max', 'Flow IAT Mean', 'Flow IAT Min', 'Max Packet Length', 'Packet Length Mean',
            'Packet Length Total', 'dstport', 'ip.len', 'ip.ttl', 'srcport', 'tcp.flags.rst', 'tcp.hdr_len',
            'tcp.window_size_value', 'udp.length']

train_data_folder_path = '/home/ddeandres/UNSW_PCAPS/train/train_data_hybrid'
test_data_folder_path = '/home/ddeandres/UNSW_PCAPS/test/csv_files'
flow_counts_file_path = '/home/ddeandres/UNSW_PCAPS/hyb_code/16-10-05-flow-counts.csv'
inference_point_n = 3
n_tree = 1

def get_x_y_flow(Dataset, feats):
    X = Dataset[feats]
    y = Dataset['Label_NEW'].replace(classes, range(len(classes)))
    sample_nature = Dataset['sample_nature']
    return X, y, sample_nature

def prepare_data(npkts, classes_filter=None):
    def assign_sample_nature(row):
        """Aux function to check the conditions and assign values"""
        if (row["Min Packet Length"] == -1 and
                row["Max Packet Length"] == -1 and
                row["Flow IAT Min"] == -1 and
                row["Flow IAT Max"] == -1):
            return "pkt"
        else:
            return "flw"

    # Load Train and Test data
    train_data = pd.read_csv(f"{train_data_folder_path}/train_data_{npkts}.csv")
    test_data = pd.read_csv(f"{test_data_folder_path}/16-10-05.pcap.txt_{npkts}_pkts.csv")
    if classes_filter is not None:
        train_data = train_data.loc[train_data['Label'].isin(classes_filter)]
        test_data = test_data.loc[test_data['Label'].isin(classes_filter)]

    flow_pkt_counts = pd.read_csv(flow_counts_file_path)

    flow_count_dict = flow_pkt_counts.set_index("flow.id")["count"].to_dict()
    # Map the values from flow_pkt_counts to test_data based on the "Flow ID" column
    test_data["pkt_count"] = test_data["Flow ID"].map(flow_count_dict)

    #### To get packet count of each flow in train data
    packet_data = pd.read_csv(f'{train_data_folder_path}/UNSW_train_ALL_PKT_DATE.csv')
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

    train_data['Label_NEW'] = np.where((train_data['Label'].isin(classes)), train_data['Label'], 'Other')
    test_data['Label_NEW'] = np.where((test_data['Label'].isin(classes)), test_data['Label'], 'Other')

    train_data['sample_nature'] = train_data.apply(assign_sample_nature, axis=1)
    test_data['sample_nature'] = test_data.apply(assign_sample_nature, axis=1)

    train_data['weight'] = np.where(train_data['sample_nature'] == 'flw',
                                    (train_data['pkt_count'] - npkts + 1) / train_data['pkt_count'],
                                    1 / train_data['pkt_count'])
    return train_data, test_data

def get_test_labels(test_data):
    classes_df = pd.DataFrame(classes, columns=['class'])
    array_of_indices = []
    unique_labels = test_data["Label_NEW"].unique()
    for lab in unique_labels:
        index = classes_df[classes_df['class'] == lab].index.values[0]
        array_of_indices.append(index)
    return unique_labels, array_of_indices

if __name__ == '__main__':
    train_data, test_data = prepare_data(inference_point_n, classes)
    test_labels, test_indices = get_test_labels(test_data)
    weight_of_samples = list(train_data['weight'])
    X_train, y_train, sample_nat_train = get_x_y_flow(train_data, features)
    X_test, y_test, sample_nat_test = get_x_y_flow(test_data, features)

    sample_ids = sample(range(X_train.shape[0]), 2000)
    X_shap_sample = X_train.iloc[sample_ids]
    y_shap_sample = y_train.iloc[sample_ids]

    model = RandomForestClassifier(n_estimators=n_tree, max_leaf_nodes=max_leaf, n_jobs=10,
                                       random_state=42, bootstrap=False)

    model.fit(X_train[features], y_train, sample_weight=weight_of_samples)
    y_pred = model.predict(X_test[features])

    y_test = [int(label) for label in y_test.values]
    y_pred = [int(label) for label in y_pred]

    class_report = classification_report(y_test, y_pred, labels=test_indices, target_names=test_labels,
                                         output_dict=True)
    print(class_report)

    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap_sample, y=y_shap_sample)
    summary_plot(shap_values, X_shap_sample)
    plt.show()
