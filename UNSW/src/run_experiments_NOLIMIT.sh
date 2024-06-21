#!/bin/bash
# N=2
echo Running Experiments:
python3 /home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/UNSW_model_analysis_FL_N5_noLimit.py "/home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/model_analysis_results/no_limit_solutions/N5_T1-10.csv" [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] [1,2,3,4,5,6,7,8,9,10] &
python3 /home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/UNSW_model_analysis_FL_N5_noLimit.py "/home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/model_analysis_results/no_limit_solutions/N5_T11-20.csv" [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] [11,12,13,14,15,16,17,18,19,20] &
python3 /home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/UNSW_model_analysis_FL_N5_noLimit.py "/home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/model_analysis_results/no_limit_solutions/N5_T21-30.csv" [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] [21,22,23,24,25,26,27,28,29,30] &
python3 /home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/UNSW_model_analysis_FL_N5_noLimit.py "/home/nds-admin/UNSW_PCAPS/hyb_code/with_FL_Metric/model_analysis_results/no_limit_solutions/N5_T31-40.csv" [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] [31,32,33,34,35,36,37,38,39,40] &
