# Data Generation
Data preparation for the ML model that targets hybrid packet- and flow-level classification.
For this you can use the python programs and packages under the `src` folder.

## Getting Started
All configurable parameters are provided via the `params.ini` file.
If you want to experiment, this is the only file you should modify.

### Configurable Parameters
The default block has a single parameter controlling the use case. Two values are possible for this parameter: `UNSW` or
`TON-IOT`. The value of the use_case parameter will decide which block will be parsed for the main parameters of the use
case.
- `data_type`: the data type specifying if the data will be used for testing or training the model, e.g., `test`. Required for naming the files.
- `label_data`: the location of the file used to label data, e.g., `/nas_storage/shared/UNSW_PCAPS/iot_device_list.csv`
- `data_path`: the location of the input pcap traces
- `inference_point_list`: the list of inference points to be used in model analysis

### What do we generate?
**Requirements:** The dataset needs to include train and test splits, and each split must have its corresponding PCAP traces.

1) **Input:** We take the pcap traces for the given split as an input to dataGenerator.
2) **Raw Packet Data Extraction:** From the pcap trace, we generate TXT file containing raw packet data along with packet-level features.
3) **Labeled Packet Data Creation:** From the TXT file, we generate a CSV file containing labeled packet data. Each packet is assigned a flow ID.
4) **Experiment Data Generation:** Using the labeled packet data, we generate the experiment dataset for hybrid packet- and flow-level classification, considering each inference point separately.
5) **Parallel Processing:** If multiple PCAP traces are provided, steps 1-4 are executed in parallel for each trace.
6) **Merging Data from Multiple Traces:** We finally merge multiple experiment data generated from different pcap traces for the same inference point.
7) **Flow Length Calculation:** We generate a separate file containing the flow length (i.e., number of packets) for each flow.

**IMPORTANT NOTE:** Flows with the same flow ID are considered unique if they originate from different PCAP traces.

### Running the analysis
After configuring the `params.ini` file you can trigger the analysis by:
```bash
source ./bin/activate
python3 generate_data.py
```
You will see logs with the progress of the execution.
