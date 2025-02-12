# Unconstrained ML model training & Obtaining feature importance values per class
Used to train an unconstrained ML model and extract the relationships between input features and output variables, per class feature importance (PCFI).
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
- `results_dir_path`: the directory to store all the results
- `inference_point_list`: the list of inference points to be explored during model analysis
- `features_set`: the list of features to be used

### Running the analysis
After configuring the `params.ini` file you can trigger the analysis by:
```bash
source ./bin/activate
python3 run_unconstrained_model_analysis.py
```
You will see logs with the progress of the execution.
