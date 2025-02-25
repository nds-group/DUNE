import ast
import configparser
from SPP.spp import SPP

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
    start_idx = int(config['CORRELATION ANALYSIS']['start_idx'])
    end_idx = int(config['CORRELATION ANALYSIS']['end_idx'])

    spp = SPP(n_classes, n_features, weights_file, f1_file, unwanted_classes)

    for i in range(start_idx, end_idx):
        file_name = f'{n_classes}CL_{i}_SPP_solution.csv'
        spp.generate_random_spp_solution(n_classes).to_csv(f'{experiment_folder_path}/{file_name}')
        print(f'Random Solution {i} at: {experiment_folder_path}/{file_name}')
    print(f'Generated {end_idx-start_idx} files.')
