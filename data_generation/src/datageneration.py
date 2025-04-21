from collections import defaultdict

import numpy as np
import pandas as pd
import subprocess
import glob
from dataclasses import dataclass
from typing import List

flow_pkt_feature_map = {'Min Packet Length': 'ip.len', 'Max Packet Length': 'ip.len',
                           'Packet Length Mean': 'ip.len', 'Packet Length Total': 'ip.len',
                           'UDP Len Min': 'udp.length', 'UDP Len Max': 'udp.length',
                           'Flow IAT Min': 'flow_iat', 'Flow IAT Max': 'flow_iat',
                           'Flow IAT Mean': 'flow_iat', 'Time to Inference': 'frame.time_relative',
                           'SYN Flag Count': 'tcp.flags.syn', 'ACK Flag Count': 'tcp.flags.ack',
                           'PSH Flag Count': 'tcp.flags.push', 'FIN Flag Count': 'tcp.flags.fin',
                           'RST Flag Count': 'tcp.flags.reset', 'ECE Flag Count': 'tcp.flags.ecn'}
flow_feature_names = list(flow_pkt_feature_map.keys())
flow_time_feature_names = ['Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean', 'Time to Inference']
flow_op_feature_map = {'Min Packet Length': 'min', 'Max Packet Length': 'max',
                           'Packet Length Mean': 'mean', 'Packet Length Total': 'sum',
                           'UDP Len Min': 'min', 'UDP Len Max': 'max',
                           'Flow IAT Min': 'min', 'Flow IAT Max': 'min',
                           'Flow IAT Mean': 'mean', 'Time to Inference': 'ptp',
                           'SYN Flag Count': 'sum', 'ACK Flag Count': 'sum',
                           'PSH Flag Count': 'sum', 'FIN Flag Count': 'sum',
                           'RST Flag Count': 'sum', 'ECE Flag Count': 'sum'}




@dataclass
class DataGenerationConfig:
    use_case: str
    data_type: str  # The type of data being generated. Either 'test' or 'train'
    labels_path: str  # The path to the file containing the labels map
    data_path: str  # The path to the data location
    inference_points_list: List[int]  # A list of inference points

    def __post_init__(self):
        # Validate that data_type must be 'test' or 'train'
        if self.data_type not in {'test', 'train'}:
            raise ValueError(f"Invalid data_type '{self.data_type}'. Must be 'test' or 'train'.")



def parse_tshark_csv_to_dataframe(filename_in):
    """
    Parses the output from tshark into a pandas DataFrame, allowing analysis of packet
    data and simplifying further processing.

    This function reads the input file containing packet capture data, processes it
    by selecting specified columns, applies appropriate data types, and calculates
    new features needed for flow identification. The resulting dataframe
    offers a structured representation for further processing or analysis.

    Parameters:
    filename_in: str
        Path to the input file containing tshark tool output in CSV format.

    Returns:
    pd.DataFrame
        A processed DataFrame where columns represent selected packet information.
        This DataFrame includes derived features such as source/destination ports
        and flow identifiers.

    Raises:
    FileNotFoundError
        If the input file specified by filename_in does not exist.
    ValueError
        If the input file is not formatted as expected to match selected columns or
        data types.
    """

    dtypes=defaultdict(lambda: 'Int64')
    dtypes["ip.src"] = "str"
    dtypes["ip.dst"] = "str"
    dtypes["eth.src"] = "str"
    dtypes["eth.dst"] = "str"
    dtypes["frame.time_relative"] = "float"

    # ToDo: this hardcodes the pcap columns. Make this configurable
    packet_data_cols = ["frame.time_relative","ip.src","ip.dst","tcp.srcport","tcp.dstport","ip.len",
                    "tcp.flags.syn", "tcp.flags.ack", "tcp.flags.push", "tcp.flags.fin",
                    "tcp.flags.reset", "tcp.flags.ecn","ip.proto", "udp.srcport", "udp.dstport",
                    "eth.src","eth.dst", "ip.hdr_len", "ip.tos", "ip.ttl", "tcp.window_size_value",
                    "tcp.hdr_len", "udp.length"]

    packet_data = pd.read_csv(filename_in, usecols=packet_data_cols, dtype=dtypes).fillna(0)

    # divide by 4, since the tcp_offset counts 32-bit words
    packet_data["tcp.hdr_len"] = packet_data["tcp.hdr_len"]/4

    packet_data["srcport"] = np.where(((packet_data["ip.proto"] == 6)), packet_data["tcp.srcport"], packet_data["udp.srcport"])
    packet_data["dstport"] = np.where(((packet_data["ip.proto"] == 6)), packet_data["tcp.dstport"], packet_data["udp.dstport"])

    packet_data = packet_data.drop(["tcp.srcport","tcp.dstport","udp.srcport","udp.dstport"],axis=1)
    packet_data = packet_data.reset_index(drop=True)

    packet_data["Flow ID"] = packet_data[["ip.src", "ip.dst", "srcport", "dstport", "ip.proto"]].astype(str).agg(
        " ".join,
        axis=1)

    return packet_data

def label_packet_data(packet_data: pd.DataFrame, config: DataGenerationConfig):
    """
    Labels the packet data based on the specified configuration and use case. The function assigns
    labels derived from the given label data to the packet data and filters the data accordingly
    based on the use case configuration.

    Args:
        packet_data (pd.DataFrame): The DataFrame containing the packet data that needs to be labeled.
        config (DataGenerationConfig): Configuration object containing paths and parameters for
            data labeling, including `labels_path` and `use_case`.

    Returns:
        pd.DataFrame: The labeled and filtered packet data.
    """
    label_data = pd.read_csv(config.labels_path)

    if config.use_case == "UNSW":
        packet_data["Label"] = [0] * len(packet_data)
        for i in range(len(label_data)):
            packet_data["Label"] = np.where((packet_data["eth.src"]==label_data["MAC ADDRESS"][i]),
                                                label_data["List of Devices"][i], packet_data["Label"])
        for i in range(len(label_data)):
            packet_data["Label"] = np.where((packet_data["eth.dst"] ==label_data["MAC ADDRESS"][i]) &
                                            (packet_data["eth.src"]=="14:cc:20:51:33:ea"),
                                            label_data["List of Devices"][i], packet_data["Label"])

        packet_data = packet_data[packet_data['Label']!="TPLink Router Bridge LAN (Gateway)"]
        packet_data = packet_data[packet_data['Label']!="0"]
        packet_data = packet_data[packet_data['Label']!="Nest Dropcam"]
        packet_data = packet_data[packet_data['Label']!="MacBook/Iphone"]
        packet_data = packet_data.reset_index(drop=True)

    if config.use_case == "TON-IOT":
        flow_list = label_data['Flow ID'].to_list()
        # Keep only the target traffic
        packet_data = packet_data[packet_data['Flow ID'].isin(flow_list)]
        flow_id_label_dict = label_data.set_index("Flow ID")["Label"].to_dict()
        # Map the labels from flow_id_label_dict to packet_data based on the "Flow ID" column
        packet_data["Label"] = packet_data["Flow ID"].map(flow_id_label_dict)

    return packet_data

def process_pcap_to_csv(pcap_file, output_file):
    """
    Processes a PCAP file and exports relevant extracted fields into a CSV-like formatted text file.

    Parameters
    ----------
    pcap_file_name : str
        The name of the PCAP file to be processed.

    Raises
    ------
    subprocess.CalledProcessError
        If the external `tshark` command fails or encounters an error during execution.

    Notes
    -----
    The function extracts specific fields from the given PCAP file using the `tshark` tool. Filtered data, excluding
    ICMP traffic but including TCP and UDP traffic, is written to an output file located in a subdirectory of `self.data_path`.
    Configuration of the fields and their format is managed by the `tshark` command options provided.
    """
    command = [
        "tshark", "-r", pcap_file, "-Y", "not icmp and (tcp or udp)", "-T", "fields",
        "-e", "frame.time_relative", "-e", "ip.src", "-e", "ip.dst",
        "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "ip.len",
        "-e", "tcp.flags.syn", "-e", "tcp.flags.ack", "-e", "tcp.flags.push",
        "-e", "tcp.flags.fin", "-e", "tcp.flags.reset", "-e", "tcp.flags.ecn",
        "-e", "ip.proto", "-e", "udp.srcport", "-e", "udp.dstport",
        "-e", "eth.src", "-e", "eth.dst", "-e", "ip.hdr_len", "-e", "ip.tos",
        "-e", "ip.ttl", "-e", "tcp.window_size_value", "-e", "tcp.hdr_len", "-e", "udp.length",
        "-E", "separator=,", "-E", "quote=n", "-E", "header=y"
    ]

    with open(output_file, "w") as out_file:
        subprocess.run(command, stdout=out_file, check=True)

def get_flow_length(data_config: DataGenerationConfig):
    """
    Fetch and process the packet data (CSV) files to calculate each flow's length.

    This function extracts the columns ('Flow ID' and 'ip.len'),
    computes the count of occurrences for each 'Flow ID', and adds additional metadata such as the file source.
    It then combines the processed data from all files into a single DataFrame.

    Parameters:
        data_config (DataGenerationConfig): Configuration object

    Returns:
        pd.DataFrame: A DataFrame consolidating information on each unique 'Flow ID'.
            The DataFrame contains:
            - 'Flow ID': Identifier of network flows.
            - 'ip.len': Length of the IP packets in each flow.
            - 'count': Number of occurrences of each 'Flow ID' in the source file.
            - 'File': Filename from which the data is extracted.

    Raises:
        FileNotFoundError: If the provided directory in `data_config.data_path` does
            not exist or contains no CSV files.
        ValueError: If an error occurs during data concatenation due to mismatched schema
            across the files.
    """
    # Create a list to store the data frames
    flow_length_dfs = []
    filenames = glob.glob(f"{data_config.data_path}/csv_files/*.csv")
    for filename in filenames:
        df = pd.read_csv(filename, usecols=['Flow ID', 'ip.len'])
        # Count occurrences of each category and add as a new column
        df['count'] = df.groupby('Flow ID')['Flow ID'].transform('count')
        df['File'] = filename.split('/')[-1].split('.')[0]
        df = df.drop_duplicates(subset=['Flow ID'])
        flow_length_dfs.append(df)

    flow_length_df = pd.concat(flow_length_dfs)

    return flow_length_df

def save_merged_data(data_config: DataGenerationConfig):
    """
    Save merged data from multiple CSV files into a single CSV file.

    The function processes data files matching a specific pattern, combines them into a single
    dataframe for each inference point value, and then saves the merged data into a new file.

    Parameters:
        data_config (DataGenerationConfig): Configuration object containing parameters such as
            the data path, inference points list, use case, and data type.

    Raises:
        None
    """
    # Iterate over the values of N
    for n in data_config.inference_points_list:
        dfs = []
        filenames = glob.glob(f"{data_config.data_path}/hybrid_data/*_{n}.csv")
        for filename in filenames:
            df = pd.read_csv(filename)
            dfs.append(df)

        merged_df = pd.concat(dfs)

        # Save the merged data frame to a CSV file
        output_filename = f"{data_config.data_path}/hybrid_data/{data_config.use_case}_{data_config.data_type}_{n}.csv"
        merged_df.to_csv(output_filename, index=False)

def generate_hybrid_data(packet_data, csv_file_name, n):
    """
    Generate hybrid features for packet data by grouping and processing specified columns.

    The function processes input packet data by computing flow features over the n first packets for each flow. Then,
    it assigns these features to consecutive packets. Additionally, it handles conversions of time-related features.

    Parameters:
        packet_data: pandas.DataFrame
            The input DataFrame containing packet-related data with columns necessary for
            feature computations (e.g., 'Flow ID', 'frame.time_relative').
        csv_file_name: str
            The name of the CSV file associated with the processed packet data.
        n: int
            The inference point over which to compute flow features.

    Returns:
        pandas.DataFrame: The updated DataFrame with hybrid features and the associated file name.
    """

    # Sort by group key and time
    packet_data = packet_data.sort_values(by=['Flow ID', 'frame.time_relative'])

    # Compute the packet number
    packet_data['pkt_number'] = (
            packet_data
            .groupby('Flow ID')  # Group by the 'key' column
            .cumcount() + 1  # Create a cumulative count starting at 1
    )

    # Compute flow_iat
    packet_data['flow_iat'] = packet_data.groupby('Flow ID')['frame.time_relative'].diff().fillna(0)

    packet_data.loc[packet_data['pkt_number'] < n, flow_feature_names] = -1

    # count_feature_names = (feature for feature in flow_feature_names if 'Count' in feature)
    for feature in flow_feature_names:
        op = flow_op_feature_map[feature]
        # handle the ptp case
        if op == 'ptp':
            op = np.ptp

        # Step 1: Compute the group-wise sum of 'tcp.flags.syn' for rows where 'pkt_number' < n
        group_aggregations = (
            packet_data[packet_data['pkt_number'] <= n]
            .groupby('Flow ID')[flow_pkt_feature_map[feature]]
            .agg(op)
        )

        # Step 2: Map the group sums back to rows where 'pkt_number' >= n
        packet_data.loc[
            packet_data['pkt_number'] >= n,
            feature
        ] = packet_data.loc[
            packet_data['pkt_number'] >= n, 'Flow ID'
        ].map(group_aggregations)

        # Step 3: convert time features to ns
        if feature in flow_time_feature_names:
            packet_data.loc[packet_data['pkt_number'] >= n, feature] = (
                    packet_data.loc[packet_data['pkt_number'] >= n, feature] * 10e9).round(9)

    packet_data = packet_data.assign(File=csv_file_name)

    return packet_data
