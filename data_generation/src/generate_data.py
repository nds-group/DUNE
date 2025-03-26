import pandas as pd
import numpy as np
import sys
import ast
import configparser
import os
from os import path
import pandas as pd
import logging
from dataGenerator import DataGenerator
import multiprocessing as mp
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

def __run_data_generation(filename, data_generator):
    logger.info(f"Starting the data generation for File: {filename}")

    data_generator.convert_pcap_to_txt(filename)
    data = data_generator.convert_txt_to_packet_data(filename)
    data_generator.generate_data(data, filename)


def main():
    data_generator = DataGenerator(use_case, data_type, inference_points_list, data_path, label_data, logger)
    
    os.makedirs(f"{data_generator.data_path}/txt_files", exist_ok=True)
    os.makedirs(f"{data_generator.data_path}/csv_files", exist_ok=True)
    os.makedirs(f"{data_generator.data_path}/hybrid_data", exist_ok=True)
    pcap_files = [f for f in os.listdir(f"{data_generator.data_path}") if ((f.endswith('.pcap') | (f.endswith('.pcapng'))))]

    __run_data_generation(pcap_files[0], data_generator)
    # consumed_cores = min([max_usable_cores, len(pcap_files)])
    # logger.info(f'Will use {consumed_cores} cores. Starting pool...')
    # with mp.get_context('fork').Pool(processes=consumed_cores) as pool:
    #     input_data = list(product(pcap_files, [data_generator]))
    #     input_data.append(data_generator)
    #     try:
    #         # issue tasks to the process pool
    #         pool.imap_unordered(run_data_generation, input_data, chunksize=chunksize)
    #         # shutdown the process pool
    #         pool.close()
    #     except KeyboardInterrupt:
    #         logger.error("Caught KeyboardInterrupt, terminating workers")
    #         pool.terminate()
    #     # wait for all issued task to complete
    #     pool.join()

    data_generator.get_flow_length()
    data_generator.merge_data()
    
    # del pool
    
    logger.info(f"Finished deta genaration, Data at: {data_generator.data_path}")

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
    inference_points_list = ast.literal_eval(config[use_case]['inference_point_list'])
    
    raise SystemExit(main())
    
