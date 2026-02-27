import numpy as np

# --- Server Power Model Parameters
P_IDLE = 50.0   # Watts (idle)
P_MAX  = 250.0  # Watts (peak)
ALPHA  = 2.0    # convexity exponent

def estimate_node_power(node_cpu_util_fraction: float) -> float:
    """
    Paper Eq.(3):
      P_node(u) = P_idle + (P_max - P_idle) * u^alpha
    where u is normalized CPU utilization in [0,1].
    """
    u = float(np.clip(node_cpu_util_fraction, 0.0, 1.0))
    return P_IDLE + (P_MAX - P_IDLE) * (u ** ALPHA)

def attribute_pod_power(
    pod_cpu_usage_cores: float,
    total_node_cpu_usage_cores: float,
    node_power: float,
    num_pods: int
) -> float:
    """
      P_pod_group = (c_group / sum_j c_j) * (P_node - P_idle) + (P_idle / N_t)

    Here we attribute:
    - the dynamic part of node power proportionally by CPU share
    - the idle part evenly across pods (N_t = num_pods)
    """
    total_node_cpu_usage_cores = float(total_node_cpu_usage_cores)
    pod_cpu_usage_cores = float(pod_cpu_usage_cores)
    num_pods = int(num_pods)

    if total_node_cpu_usage_cores <= 0.0:
        return 0.0

    # Dynamic portion (node_power - P_IDLE), clamp to avoid negative due to noise
    dynamic_power = max(0.0, float(node_power) - P_IDLE)

    # CPU share of the pod group (clamp to [0,1] for robustness)
    pod_share = pod_cpu_usage_cores / total_node_cpu_usage_cores
    pod_share = float(np.clip(pod_share, 0.0, 1.0))
    pod_dynamic_power = dynamic_power * pod_share

    # Idle portion distributed evenly across active pods
    if num_pods <= 0:
        pod_idle_power_share = 0.0
    else:
        pod_idle_power_share = P_IDLE / num_pods

    return pod_dynamic_power + pod_idle_power_share
