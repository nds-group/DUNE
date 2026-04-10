# DUNE Tofino Implementation

This directory contains the P4 source code, table entry generators, and control plane scripts required to deploy the DUNE framework on Intel Tofino hardware switches.

**⚠️ IMPORTANT WARNING:**
The code generation in this directory is target-specific (Intel Tofino) and use-case specific (ToN-IoT and UNSW). As such, the generation of the P4 code and control-plane scripts has **not been automated**. The scripts contain hardcoded paths that you must modify before executing them in your local environment.

## Repository Structure

The directory is divided by use case and their respective ML inference sub-models (Clusters):
- `ToN-IoT/`: Contains the implementation for the ToN-IoT dataset.
  - Subdirectories (e.g., `CL1`, `CL3`, `CL0-2`) correspond to the specific ML cluster implementations.
- `UNSW/`: Contains the implementation for the UNSW dataset.
  - Subdirectories (e.g., `CL1-3`, `CL2-0`, `CL4-5`) correspond to the specific ML cluster implementations.

## Execution Workflow

To run a specific cluster on the Intel Tofino switch, you must follow these steps in your SDE environment.

### 1. Compile the P4 Program
Compile the cluster's `.p4` source code using the Intel P4 Studio (SDE).
```bash
~/tools/p4_build.sh <cluster_program>.p4
```

### 2. Generate Table Entries
The machine learning sub-models are represented as `.sav` pickle files. You need to convert them into match-action table entries for the switch.
1. Open the `generate_te_*.py` script in the cluster directory.
2. **Modify the hardcoded `clf = pd.read_pickle(...)` path** to point to the local `.sav` file provided in the same directory.
3. Run the script:
   ```bash
   python3 generate_te_*.py
   ```
   *This will output a Python script (e.g., `te_XXX.py`) that contains the BfRt table programming commands.*

### 3. Start the Switch Daemon
Start the `bf_switchd` process to load the compiled P4 program onto the Tofino ASIC:
```bash
$SDE_INSTALL/bin/run_switchd.sh -p <cluster_program>
```

### 4. Populate the Tables
In a separate terminal, open the BfRt Python CLI and execute the generated table entries script to program the data plane:
```bash
$SDE/run_bfshell.sh -b te_XXX.py
```
*(Alternatively, you can run `exec(open('te_XXX.py').read())` inside `bfrt_python`)*

### 5. Run the Control Plane (Digest) Script
To collect inference results and handle the hybrid classification, run the `controller_digest_hybrid.py` script:
```bash
python3 controller_digest_hybrid.py output_results.csv
```

## Running Sequence

If you intend to run the full distributed pipeline across the clusters, they must process the traffic in the correct sequence. 

For **ToN-IoT**, the required sequential order of execution across the clusters is:
1. `CL1`
2. `CL3`
3. `CL0-2`
