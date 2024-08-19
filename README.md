# DUNE: Distributed Inference in the User Plane
This repository contains the source code of DUNE
## Abstract
The deployment of Machine Learning (ML) models in the user plane, enabling line-rate in-network inference, significantly
reduces latency and improves the scalability of cases like traffic monitoring.
Yet, integrating ML models into programmable network devices requires meeting stringent constraints in terms of memory
resources and computing capabilities.
Previous solutions have focused on implementing monolithic ML models within individual programmable network devices,
which are limited by hardware constraints, especially while executing challenging classification use cases.
In this paper, we propose `DUNE`, a novel framework that realizes for the first time a user plane inference that is
distributed across the multiple devices that compose the programmable network. `DUNE` adopts fully automated approaches
to (i) breaking large ML models into simpler sub-models that preserve inference accuracy while minimizing
resource usage, (ii) designing the sub-models and their sequencing to enable an efficient distributed
execution of joint packet- and flow-level inference.
We implement `DUNE` using P4, deploy it in an experimental network with multiple industry-grade programmable switches,
and run tests with real-world traffic measurements for two complex inference use cases. Our results demonstrate that
`DUNE` not only reduces per-switch resource utilization with respect to legacy monolithic ML designs but also improves
their inference accuracy by up to 7.5%.

## Repository Structure
Each step in the workflow is separated into its own folder.
Inside each folder, you will find a `README.md` file with instructions on how to run each step.
### Unconstrained ML model training
Used to train an unconstrained ML model and extract the relationships between input features and output variables,
per class feature importance (PCFI).
### Model Paritioning
Used to break down the original inference task into a series of smaller sub-tasks (cluster) that jointly achieve
the same goal.
### Cluster Analysis
Used to evaluate the F1 Score of a given solution
### Model Sequencing
used to order the ML sub-models to optimize the inference performance.