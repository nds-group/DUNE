# Model Sequencing
To find a better sequence of sub-models, we define and solve a Travelling Salesman Problem by formulating as an Integer Linear Programming.
For this you can use the python programs and packages under the `src` folder.

## Getting Started
All configurable parameters are provided via the `params.ini` file.
If you want to experiment, this is the only file you should modify.

### Configurable Parameters
The default block has a single parameter controlling the use case. Two values are possible for this parameter: `UNSW` or
`TON-IOT`. The value of the use_case parameter will decide which block will be parsed for the main parameters of the use
case.
- `classes_filter`: A list with the subset of classes you wish to use.
- `train_data_dir_path`: the location of the training data, e.g., `/home/beyzabutun/shared/ToN-IoT/data/train`
- `test_data_dir_path`: the location of the test data
- `flow_counts_test_file_path`: the path to the csv file with the flow counts for the test set, e.g., `/home/ddeandres/ToN-IoT/ToN_IoT_Test_Flow_PktCounts.csv`
- `flow_counts_train_file_path`: the path to the csv file with the flow counts for the training set.
- `best_models_per_cluster_path`: the path to the csv with the selected model information per cluster

### Running the analysis
After configuring the `params.ini` file you can trigger the analysis by:
```bash
source ./bin/activate
python3 model_sequencing.py
```
You will see logs with the progress of the execution.
