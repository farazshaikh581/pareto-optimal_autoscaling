import numpy as np

# --- Server Power Model Parameters ---
# Based on typical server specifications and cited research models.
P_IDLE = 50  # Idle power in Watts for a typical server
P_MAX = 250  # Max power at 100% utilization in Watts
ALPHA = 2.0  # Exponent for non-linear power curve, as seen in square models [3]

def estimate_node_power(node_cpu_util_fraction):
    """Estimates total node power based on CPU utilization using a non-linear model."""
    dynamic_power = (P_MAX - P_IDLE) * (node_cpu_util_fraction ** ALPHA)
    return P_IDLE + dynamic_power

def attribute_pod_power(pod_cpu_usage_cores, total_node_cpu_usage_cores, node_power, num_pods):
    """Attributes a portion of the node's power to a group of pods based on their CPU share."""
    if total_node_cpu_usage_cores == 0:
        return 0
    
    # Attribute dynamic power proportionally to CPU usage, inspired by Kepler's ratio model [4]
    dynamic_power = node_power - P_IDLE
    pod_share_of_dynamic = pod_cpu_usage_cores / total_node_cpu_usage_cores
    pod_dynamic_power = dynamic_power * pod_share_of_dynamic
    
    # Distribute the static idle power evenly among all running pods
    pod_idle_power_share = (P_IDLE / 100) * num_pods # Assuming a max of 100 pods for idle share distribution
    
    return pod_dynamic_power + pod_idle_power_share
