import configparser
import logging
import os
import multiprocessing as mp
from datageneration import *
from os import path
from ast import literal_eval
from itertools import product
                
def run_data_generation(input_data):
    filename = input_data[0]
    data_generator_obj = input_data[1]
    try:
        return __run_data_generation(filename, data_generator_obj)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []

def __run_data_generation(file: str, config: DataGenerationConfig):
    logger.info(f"Starting the data generation for File: {file}")


    pcap_file_path = os.path.join(config.data_path, file)
    pcap_file_name = file.split('.')[0]
    csv_file_name = f"{pcap_file_name}.csv"

    tshark_file_path = os.path.join(config.data_path, 'tshark_files', f"{pcap_file_name}.csv")
    hybrid_results_dir = f"{config.data_path}/hybrid_data"

    process_pcap_to_csv(pcap_file_path, tshark_file_path)
    packet_data = parse_tshark_csv_to_dataframe(tshark_file_path)

    labeled_packet_data = label_packet_data(packet_data, config)
    labeled_packet_data.to_csv(f"{data_path}/csv_files/{pcap_file_name}.csv", index=False)


    output_columns = ['Flow ID', 'ip.len', 'ip.ttl', 'tcp.flags.syn', 'tcp.flags.ack', 'tcp.flags.push',
                      'tcp.flags.fin', 'tcp.flags.reset', 'tcp.flags.ecn', 'ip.proto', 'srcport', 'dstport',
                      'ip.hdr_len', 'ip.tos', 'tcp.window_size_value', 'tcp.hdr_len', 'udp.length',
                      'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Total',
                      'UDP Len Min', 'UDP Len Max', 'Flow IAT Min', 'Flow IAT Max', 'Flow IAT Mean',
                      'Time to Inference', 'SYN Flag Count', 'ACK Flag Count', 'PSH Flag Count', 'FIN Flag Count',
                      'RST Flag Count', 'ECE Flag Count', 'Label', 'File']

    for n in config.inference_points_list:
        packet_data = generate_hybrid_data(packet_data, csv_file_name, n)

        filename_out = f"{hybrid_results_dir}/{config.use_case}_{config.data_type}_{pcap_file_name}_N_{n}.csv"
        packet_data.to_csv(filename_out, columns=output_columns, index=False)


def main():
    data_config = DataGenerationConfig(
        use_case=use_case,
        data_type=data_type,
        labels_path=label_data,
        data_path=data_path,
        inference_points_list=inference_points_list
    )

    os.makedirs(f"{data_path}/tshark_files", exist_ok=True)
    os.makedirs(f"{data_path}/csv_files", exist_ok=True)
    os.makedirs(f"{data_path}/hybrid_data", exist_ok=True)
    pcap_files = [f for f in os.listdir(f"{data_path}") if ((f.endswith('.pcap') | (f.endswith('.pcapng'))))]

    consumed_cores = min([max_usable_cores, len(pcap_files)])
    logger.info(f'Will use {consumed_cores} cores. Starting pool...')
    with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
        input_data = list(product(pcap_files, [data_config]))
        try:
            # issue tasks to the process pool
            pool.imap_unordered(run_data_generation, input_data, chunksize=chunksize)
            # shutdown the process pool
            pool.close()
        except KeyboardInterrupt:
            logger.error("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
        # wait for all issued task to complete
        pool.join()

    # Once all files have been processed, we can merge them into individual files and save them
    save_merged_data(data_config)

    flow_length_data = get_flow_length(data_config)
    flow_length_data.to_csv(f"{data_path}/{use_case}_flow_length.csv", index=False)

    del pool
    
    logger.info(f"Finished deta genaration, Data at: {data_config.data_path}")

if __name__ == '__main__':
    basepath = path.dirname(__file__)
    logging.basicConfig()
    config = configparser.ConfigParser()
    config.read(path.abspath(path.join(basepath,'params.ini')))
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    max_usable_cores = int(config['DEFAULT']['max_usable_cores'])
    chunksize = int(config['DEFAULT']['chunksize'])
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)

    # These are parameters for the data generation
    data_type = config[use_case]['data_type']
    label_data = config[use_case]['label_data']
    data_path = config[use_case]['data_path']
    inference_points_list = literal_eval(config[use_case]['inference_point_list'])
    
    raise SystemExit(main())
    
