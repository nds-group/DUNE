# Cluster Analysis
Once an unconstrained ML model is paritioned into smaller sub-tasks, you might want to analyse its performance.
To this end you can use the python programs and packages under the `src` folder.

## Getting Started
All configurable parameters are provided via the `params.ini` file.
Therefore, unless you want to experiment, this is the only file you should modify.
### Configurable Parameters
The `.ini` file is structured into sections.
First you find the default section with parameters with a global effect.
- `log_level`: allows you to control the logging level
- `use_case`: currently only two use cases are supported, `UNSW` or `TON-IOT`. This value also specified which of the
parameters sections will be parsed.
- `force_rewrite`: Set to `True` to overwrite stored results from a previous execution. Since analysing the models is time-consuming, specially when performing a grid search, if a running
instance gets terminated before completion, it is possible to restart from the last completed checkpoint (grid point).
Thereforem if you wish to overwrite the stored results and start from the beginning, you should set this parameter to
`True`. Otherwise, set to `False`.
- `grid_search`: when this parameter is set to `True`, it performs a time-consuming, yet exhaustive parameters search.
It produces the best results.
- `max_usable_cores`: controls how many cores are available for the multiprocessing pool.
Use carefully as you might run out of RAM.
- `chunksize`: controls the number of tasks sent to each of the cores during the multiprocessing.
- `features_filter`: a list that controls the features to be used in the analysis. Leave empty for all to be used. Make
sure that the override_features param is set to `True` at the use case level.
- 
Second you find the use case level parameters. You can define many use cases simultaneously and use the `use_case`
parameter to switch from one to another. These are:
- `classes_filter`: A list with the subset of classes you wish to use.
- `train_data_dir_path`: the location of the training data, e.g., `/home/ddeandres/ToN-IoT/data/train`
- `test_data_dir_path`: the location of the test data
- `flow_counts_test_file_path`: the path to the csv file with the flow counts for the test set, e.g., `/home/ddeandres/ToN-IoT/ToN_IoT_Test_Flow_PktCounts.csv`
- `flow_counts_train_file_path`: the path to the csv file with the flow counts for the training set.
- `cluster_data_file_path`: the path to the csv file with the cluster description, e.g., `/home/ddeandres/TON-IOT_SPP_solution_20240724_132909.csv`. This file specifies the classes and features to be used in each sub-problem.
- `results_dir_path`: the path where the analysis results should be saved. It will also be used as working directory.

### Running the analysis
After configuring the `params.ini` file you can trigger the analysis by:
```bash
source ./bin/activate
python3 run_cluster_analysis.py
```
You will see logs with the progress of the execution.

## Additional Scripts
Although not required for `DUNE`, there are additional files that may be of interest.
### Correlation Analysis
Searching for a suitable optimization model for the partitioning, you might want to evaluate many different partitions.
For this, store all the solutions in individual CSV files and store them in the same folder. Pass the location of this
folder as a parameter to the script via the `results_dir_path` parameter.
To start the execution you may use:
```bash
source ./bin/activate
python3 run_correlation_analysis.py
```
Since it might take long, you might want to run it from a tmux session. Also you might want to monitor the execution 
using htop. To check whether some process was killed due to memory issues use:
```watch -n 10 "dmesg -T | egrep -i 'killed process'"```
If some process is killed. You should kill the parent instance and restart the execution of the main program.
It will resume from the next cluster to be analysed.
### Simple classifier
To run a simple classifier over the data:
`python3 run_simple_classifier.py`