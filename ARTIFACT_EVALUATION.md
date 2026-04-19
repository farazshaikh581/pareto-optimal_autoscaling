# Artifact Evaluation Guide

This guide provides reviewers with the necessary instructions to reproduce the results and evaluate the artifacts submitted for:
**Pareto-Optimal Autoscaling: A Multi-Objective Reinforcement Learning Framework for the Performance-Cost-Energy Trilemma**

## 1. System Requirements

The framework interacts in **real-time** with a live MicroK8s cluster to perform actions. Due to this active interaction, the following environment is strictly required:

### Software Dependencies
- **OS**: Ubuntu 20.04+ (or equivalent Linux distribution supporting MicroK8s).
- **Cluster**: MicroK8s (`microk8s kubectl` command must be active).
- **Load Generator**: `hey` (https://github.com/rakyll/hey) installed and added to `$PATH`.
- **Metrics**: A Prometheus instance actively scraping Kubernetes metrics.
- **Python**: Python 3.8+ with exactly matched dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

### Hardware Resources
- **Time Constraint**: The pipeline steps execute in 60-second real-time intervals. Training a single agent instance over a "3-day simulated duration" physically requires **~72 wall-clock hours**.
- Training the entire 4-profile suite sequentially takes **~12 days of compute time**. Artifact Evaluators should be aware of this execution cost when attempting a from-scratch reproduction.

## 2. Infrastructure Setup
1. **Deploy Target Application**: Ensure `factorizator-deployment.yaml` and `RBAC.yaml` are applied to the `factorizator` namespace.
2. **Expose Services**: The Python script needs direct HTTP access to the application endpoints.
3. Update `run_simulation.py` with the environmental parameters:
   - `PROMETHEUS_URL`: Your actual Prometheus scraping endpoint.
   - `url_app_service`: The reachable target endpoint responding to GET requests.

## 3. Dataset Traces
Download the Azure Invocation Traces (2021) into the project directory:
[AzureFunctionsInvocationTrace2021.md](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md)
Rename it precisely to `AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`.

## 4. Reproducing Findings

The scripts have been designed for perfect reproducibility by strictly enforcing random seeds across `os`, `random`, `numpy`, `torch` and PyTorch internal CUDA algorithms (`torch.backends.cudnn`).

### Training Agents
You can explicitly train any of the profile objectives (e.g., `perf_focused`, `cost_focused`, `energy_focused`, `balanced`):

```bash
python run_simulation.py --mode train --profile balanced --seed 42
```
Checkpoints and normalization parameters will save directly under `./results/checkpoints/<profile>`.

### Testing and Metric Generation
To evaluate the stable checkpoints and generate the inference benchmark CSVs:

```bash
python run_simulation.py --mode test --profile balanced --seed 42
```
A successful test outputs a results matrix (`*_test_run.csv`) containing step-by-step latency, scaling decisions, energy output models, and reward traces identically mapping to the figures presented in the paper.

## 5. Artifact Plot Rendering (Offline Reproduction)
If physical cluster evaluation is prohibitive, we have included the raw results (`Results Files/`) extracted from the experiments.
You can directly parse these output CSVs and generate the exact graphs used in our paper offline by running our integrated notebook:
```bash
cd "Results Files"
jupyter notebook results_plots.ipynb
```
