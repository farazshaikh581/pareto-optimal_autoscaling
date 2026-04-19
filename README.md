# Pareto-Optimal Autoscaling
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Environment: MicroK8s](https://img.shields.io/badge/kubernetes-microk8s-326ce5.svg)](https://microk8s.io/)
[![RL: StableBaselines3](https://img.shields.io/badge/rl-stable--baselines3-brightgreen)](https://stable-baselines3.readthedocs.io/en/master/)

Official implementation of the **NOMS 2026** paper:  
> *"Pareto-Optimal Autoscaling: A Multi-Objective Reinforcement Learning Framework for the Performance-Cost-Energy Trilemma"*

---

## 📖 Abstract

Cloud-native systems, especially on serverless edge platforms, face significant operational complexity in balancing the competing objectives of application performance, operational cost, and energy consumption. This performance-cost-energy trilemma cannot be solved by traditional, single-objective autoscalers. This paper proposes a Multi-Objective Reinforcement Learning (MORL) framework based on Proximal Policy Optimization (PPO) along with a Transformer-based policy network to model long-range temporal dependencies for proactive scaling. Using a weighted-sum reward function, we train four distinct agents with reward vectors prioritizing performance, cost, energy, and a balanced approach, respectively. The experimental results, based on a realistic web-service workload, demonstrate that the framework successfully generates a set of different, non-dominated policies forming a Pareto-optimal front. We show that the  agent focused on performance learns an aggressive scaling policy minimizing latency at the expense of modest resource use, while those focused on cost and energy converge to a similar, resource-minimal strategy. This convergence reveals a high correlation between cost and energy objectives within our system. The \textit{balanced} agent successfully identifies a stable compromise policy. This MORL approach provides cloud operators with a 'menu' of optimal, pre-trained policies, enabling dynamic selection to align with changing business or sustainability goals.

## 🚀 Reproducibility Guide

### 1. Requirements

We recommend running this on an **Ubuntu 20.04+** machine with at least 4 cores and 8GB RAM due to the overhead of the live Kubernetes environment.

Ensure you have Python 3.8+ installed. Install the exact versions of the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
This framework is evaluated against real-world Microsoft Azure serverless invocation traces. Due to the scale of the dataset, it is not tracked directly in this repository.

To download the required payload:
1. Visit the [Azure Public Dataset (2021 traces)](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md).
2. Download the specific trace segment representing the two weeks of January 2021 context.
3. Save it to the root of this repository and rename it precisely to:  
   `AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt`

### 3. Kubernetes Configuration

The simulation actively issues control actions to a running MicroK8s cluster and fetches live metrics via a Prometheus endpoint. Before initiating the training, update the global variables inside `run_simulation.py`:
- `PROMETHEUS_URL`: Your Prometheus endpoint
- `url_app_service`: Your NodePort reachable app service URL

### 4. Usage

**Training**

You can train the agents for different profiles (`perf_focused`, `cost_focused`, `energy_focused`, `balanced`). The `--seed` argument ensures deterministic outcomes.

```bash
python run_simulation.py --mode train --profile balanced --seed 42
```

**Testing**

Once a model is trained, evaluate its performance across the test set:

```bash
python run_simulation.py --mode test --profile balanced --seed 42
```

Results are dynamically saved into `./results/test/<profile>` and TensorBoard metrics are logged to `./results/tb_test/`.

---

## 📊 Artifact Evaluation & Result Plotting

If you do not have the time/infrastructure to execute a 12-day run on a physical Kubernetes cluster, we have archived our experiment logs inside the `Results Files/` directory for immediate offline verification. 

You can rigorously reproduce the **3D Pareto-Front** and metric boxplots shown in our NOMS paper seamlessly:
```bash
cd "Results Files"
python generate_plots.py
```
All visual exports will identically output into the `Results Files/figures/` directory.
