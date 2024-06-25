#!/bin/bash
echo Running Experiments:
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 0 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 1 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 2 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 3 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 4 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 5 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 6 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
python3 ~/distributed_in_band/UNSW/src/UNSW_model_analysis_FL_XCLASS_0.py 7 ~/distributed_in_band/UNSW/cluster_info/UNSW_SPP_solution.csv &
