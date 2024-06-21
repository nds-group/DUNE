#!/bin/bash
# N=2
echo Running Experiments:
python3 /home/nds-admin/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withoutOther.py 1 /home/nds-admin/distributed_in_band/ToN-IoT/cluster_info/cluster_info_ToNIoT_10CL_6cluster.csv &
python3 /home/nds-admin/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withoutOther.py 2 /home/nds-admin/distributed_in_band/ToN-IoT/cluster_info/cluster_info_ToNIoT_10CL_6cluster.csv &
python3 /home/nds-admin/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withoutOther.py 3 /home/nds-admin/distributed_in_band/ToN-IoT/cluster_info/cluster_info_ToNIoT_10CL_6cluster.csv &
python3 /home/nds-admin/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withoutOther.py 4 /home/nds-admin/distributed_in_band/ToN-IoT/cluster_info/cluster_info_ToNIoT_10CL_6cluster.csv &
python3 /home/nds-admin/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withoutOther.py 5 /home/nds-admin/distributed_in_band/ToN-IoT/cluster_info/cluster_info_ToNIoT_10CL_6cluster.csv &
python3 /home/nds-admin/distributed_in_band/ToN-IoT/src/ToN-IoT_model_analysis_FL_XCLASS_withoutOther.py 6 /home/nds-admin/distributed_in_band/ToN-IoT/cluster_info/cluster_info_ToNIoT_10CL_6cluster.csv &