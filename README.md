# DUNE: Distributed Inference in the User Plane

This repository contains the source code of **DUNE**, a novel framework for distributed ML inference in the user plane.

- 📄 **[PDF pre-print of DUNE](https://dspace.networks.imdea.org/handle/20.500.12761/1883)**, published in IEEE INFOCOM 2025.
- 🏆 **[DUNE received INFOCOM 2025 (Best paper award)](https://infocom2025.ieee-infocom.org/awards#:~:text=DUNE%3A%20Distributed%20Inference%20in%20the%20User%20Plane)**

## Abstract
The deployment of Machine Learning (ML) models in the user plane, enabling line-rate in-network inference, significantly reduces latency and improves the scalability of cases like traffic monitoring. Yet, integrating ML models into programmable network devices requires meeting stringent constraints in terms of memory resources and computing capabilities.

Previous solutions have focused on implementing monolithic ML models within individual programmable network devices, which are limited by hardware constraints. In this paper, we propose `DUNE`, a novel framework that realizes for the first time user plane inference distributed across multiple programmable network devices. `DUNE` adopts fully automated approaches to (i) break large ML models into simpler sub-models that preserve inference accuracy while minimizing resource usage, and (ii) design the sub-models and their sequencing to enable an efficient distributed execution of joint packet- and flow-level inference.

## Tutorial
A comprehensive tutorial is available as a Jupyter notebook (`dune_tutorial.ipynb`). We strongly encourage users to explore this notebook as a starting point for getting acquainted with the DUNE framework and its pipeline steps.

## EXTENSION: DUNE in Software Switches (bmv2) and Arbitrary Topologies
**`dune-bmv2/` (Submodule)**: This directory is a Git submodule corresponding to a software switch (bmv2) implementation of DUNE that enables reproducing the data-plane execution of DUNE in Mininet. It includes P4 programs, Mininet scripts, and instructions for setting up the environment and running the experiments. User without Hardware switches can use this submodule to validate the DUNE framework in a software-based environment.

## Repository Structure & Pipeline Workflow
DUNE is designed as a linear pipeline. Each step in the workflow is separated into its own directory and depends on the output of the previous step. Detailed instructions and guidelines for running each step can be found in their respective `README.md` files.

1. **`data_generation/`**  
   Transforms and prepares the data from the source datasets for training and evaluating the ML models, including ground truth CSV files and PCAPs.
2. **`unconstrained_model_analysis/`**  
   Train an unconstrained ML model and extract the relationships between input features and output variables (per-class feature importance).
3. **`model_partitioning/`**  
   Break down the original inference task into a series of smaller sub-tasks (clusters) that jointly achieve the same goal.
4. **`cluster_analysis/`**  
   Evaluate the F1 Score of a given solution.
5. **`model_sequencing/`**  
   Order the ML sub-models to optimize inference performance.
6. **`dune-tofino/`**  
   Implementation of the UNSW, and ToN-IoT use cases for the Intel Tofino P4 target.  
   *Note: The generation of this code is target- and use-case specific; thus, it has not been automated.*

## Dependencies & Installation

- **Python 3.x**: Required for the ML pipeline steps.
- **Python Dependencies**: Dependencies are isolated per directory. Before running the scripts for a specific pipeline step, install its dependencies:
  ```bash
  pip install -r <directory_name>/requirements.txt
  ```
- **BMv2 & Mininet**: Required for the data-plane execution in the `dune-bmv2` submodule. Please refer to `dune-bmv2/README.md` for specific P4 compilation and Mininet setup instructions.

## Prerequisites & External Files

To reproduce the experiments in the paper, you must obtain the datasets (such as the ToN_IoT datasets or equivalent traffic traces) and place them in the locations expected by the pipeline scripts.

1. **Obtain Data:** Download the appropriate ground truth CSV files and PCAPs as detailed in the paper and the individual subdirectory `README.md` guidelines.
2. **Placement:** Ensure these files are placed inside the `data/` or `pcaps/` directories as requested by each step's instructions before starting the execution.
