#!/bin/bash
# N=2
echo Running Experiments:
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withOther.py 1 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv &
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withOther.py 2 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv &
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withOther.py 3 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv &
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withOther.py 4 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv &
