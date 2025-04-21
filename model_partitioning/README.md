# Model Partitioning
To partition a tree-based classification model into smaller sub-problems we solve a set partitioning problem.
For this you can use the python programs and packages under the `src` folder.

## Getting Started
All configurable parameters are provided via the `spp_params.ini.example` file.
You can copy the `.example` file as an initial template.
Therefore, unless you want to experiment, this is the only file you should modify.

### Configurable Parameters
The default block has a single parameter controlling the use case. Two values are possible for this parameter: `UNSW` or
`TON-IOT`. The value of the use_case parameter will decide which block will be parsed for the main parameters of the use
case.
At the use case block level you will find:
- `n_classes`: an integer specifying the number of classes to use from those available ,e.g., 24. The cut will be made
following alphabetical order.
- `n_features`: an integer specifying the number of features to use from those available ,e.g., 15. The cut will be made
following alphabetical order.
- `weights_file`: the path to the CSV file including the PCFI values for the given use case,e.g., `importance_weights.csv`
- `f1_file`: the path to the CSV file including the attained F1 score for each of the classes, e.g., `score_per_class.csv`
- `unwanted_classes`: through this list you want blacklist classes you do not wish to include, e.g., `['Blipcare Blood Pressure meter', 'IPhone']`
- `fix_level`: while the optimization problem already finds the optimal number of clusters, you may fix the number
through this parameter. Comment out fix_level if not desired. Note that the level = n_clusters + 1

### Running the analysis
After configuring the `spp_params.ini.example` file you can trigger the analysis by:
```bash
source ./bin/activate
python3 model_partitioning.py
```
You will see logs with the progress of the execution.

In addition, you can uncomment other analysis inside this file, like brute force analysis or time execution.
**Alternatively, you can use the notebook provided in the notebooks folder to easily visualize results and execute
particular functions.**
