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
1) We take the pcap traces as an input to dataGenerator.
2) From the pcap trace, we generate txt file including raw data of packets with packet-level features.
3) From the txt file, we generate labeled packet data (as .csv) - with the defined flow id per packet.
4) From the labeled packet data, we generate train/test data for each inference point for hybrid packet- and flow-level classification. 
5) If we have multiple pcap traces as an input, we run the steps 1-4 for each pcap trace in parallel.
6) We finally merge multiple train/test data generated from different pcap traces for the same inference point.
7) We generate a file including the flow length per flow.

**IMPORTANT NOTE:** The flows with the same flow id is considered as different flows if they are generated from different pcap traces. Please make sure to merge all the input pcap traces if you want these flows to be considered as the same flow.

### Running the analysis
After configuring the `params.ini` file you can trigger the analysis by:
```bash
source ./bin/activate
python3 generate_data.py
```
You will see logs with the progress of the execution.
