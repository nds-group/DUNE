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
                
def run_data_generation(input_data):
    data_generator = input_data
    try:
        return __run_data_generation(data_generator)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return []

def __run_data_generation(data_generator):
    logger.info(f"Starting the data generation...")

    # data_generator.convert_pcap_to_txt(data_generator.pcap_folder)
    
    # List all .txt files in the current directory
    pcap_files = [f for f in os.listdir(f"{data_generator.pcap_folder}/") if f.endswith('.txt')]
    for f in pcap_files:
        data_generator.convert_txt_to_packet_data(f)
        data_generator.generate_data(f)
    
    logger.info(f"Finished deta genaration, Data at: {data_generator.pcap_folder}")

def main():
    data_generator = DataGenerator(use_case, inference_points_list, pcap_folder_path, label_data, logger)
    run_data_generation(data_generator)

if __name__ == '__main__':
    basepath = path.dirname(__file__)
    logging.basicConfig()
    config = configparser.ConfigParser()
    config.read(path.abspath(path.join(basepath,'params.ini')))
    use_case = config['DEFAULT']['use_case']
    log_level = config['DEFAULT']['log_level']
    level = logging.getLevelName(log_level)
    logger = logging.getLogger(use_case)
    logger.setLevel(level)

    # These are parameters for the data generation
    label_data = config[use_case]['label_data']
    pcap_folder_path = config[use_case]['results_dir_path']
    inference_points_list = ast.literal_eval(config[use_case]['inference_point_list'])
    
    raise SystemExit(main())
    
    
    
        
        

        