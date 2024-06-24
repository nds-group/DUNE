#!/bin/bash
# N=2
echo Running Experiments:
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_Ordered.py 1 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv ~/distributed_in_band/ToN-IoT/cluster_info/cluster_order_10CL_SPP_MACRO_withOptimizer_4Cluster_2013.csv &
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_Ordered.py 2 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv ~/distributed_in_band/ToN-IoT/cluster_info/cluster_order_10CL_SPP_MACRO_withOptimizer_4Cluster_2013.csv &
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_Ordered.py 3 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv ~/distributed_in_band/ToN-IoT/cluster_info/cluster_order_10CL_SPP_MACRO_withOptimizer_4Cluster_2013.csv &
python3 ~/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_Ordered.py 4 ~/distributed_in_band/ToN-IoT/cluster_info/ToN-IoT_SPP_solution.csv ~/distributed_in_band/ToN-IoT/cluster_info/cluster_order_10CL_SPP_MACRO_withOptimizer_4Cluster_2013.csv &
