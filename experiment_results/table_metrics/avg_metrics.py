import json
import numpy as np

# Set paths
input_files = ['WRN_CIFAR10_TS.txt', 'WRN_CIFAR10.txt']
output_file = "metrics_summary.txt"


metrics_data = {}
for filename in input_files:
    with open(filename, 'r') as file:
        data = json.load(file)
        
        # Loop over each entry in the JSON structure
        for key, metrics in data.items():
            for metric, value in metrics.items():
                # If the value is a dictionary (for nested metrics like "SHIFT Intensity")
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        full_metric = f"{metric}_{sub_metric}"
                        metrics_data.setdefault(full_metric, []).append(sub_value)
                elif isinstance(value, list):
                    # If the metric is a list (like "OOD AUROC"), compute the average of the list
                    metrics_data.setdefault(metric, []).append(np.mean(value))
                else:
                    metrics_data.setdefault(metric, []).append(value)

# Compute mean and standard deviation for each metric
metrics_summary = {}
for metric, values in metrics_data.items():
    print(metric, values)
    metrics_summary[metric] = {
        "average": np.mean(values),
        "std_dev": np.std(values)
    }


with open(output_file, 'w') as output:
    json.dump(metrics_summary, output, indent=4)

print(f"Metrics summary saved to {output_file}")
