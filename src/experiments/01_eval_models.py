from src.lightning_modules.One_Stage import *
import numpy as np
from src.utils.eval_utils import *
import torch.nn as nn
#from src.utils.metrics import *
import os
from argparse import ArgumentParser
import json
import argparse
from src.utils.utils import *
import torch
from src.lightning_modules.Two_Stage import *

parser = ArgumentParser()
parser.add_argument("--save_file_name", type=str, default="")
parser.add_argument("--model_name_file", type=str, default="")
parser.add_argument("--num_samples", type=int, default=1)
parser.add_argument("--temperature_scale", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()


if args.save_file_name == "":
    raise Exception("Oops you did not provide a save_file_name!")
if args.model_name_file == "":
    raise Exception("Oops you did not provide a model_name_file!")
root_dir = "./"
batch_size=512

model_paths = open("./eval_path_files/"+args.model_name_file, "r")

if torch.cuda.is_available():
    device = 'cuda:0'
    device_auto = "gpu"
else:
    device = 'cpu'
    device_auto = "cpu"

if os.path.isfile('./experiment_results/table_metrics/'+args.save_file_name):
    f = open('./experiment_results/table_metrics/'+args.save_file_name, 'r') 
    results = json.load(f) 
else:
    results = {}

num_models = 0
for model_path in model_paths.read().splitlines():
    model_path = model_path.strip()
    model_name = model_path.split("model_name=")[1].replace(".ckpt", "")
    dataset = model_name.split("_")[1]
    model_type = model_name.split("_")[0]
    if args.temperature_scale:
        model_name = "Temp-"+model_name

    if model_name not in results.keys():
        ood_done = False
        in_done = False
        shift_done = False
        train_done = False
        results[model_name] = {}
    else:
        ood_done = True
        in_done = True
        shift_done = True
        train_done = True
        if 'clean_accuracy' not in results[model_name].keys():
            in_done = False
        if 'SHIFT ECE' not in results[model_name].keys():
            shift_done = False
        if 'OOD AUROC' not in results[model_name].keys():
            ood_done = False
        if 'Train NLL' not in results[model_name].keys():
            train_done = False
        if ood_done and in_done and shift_done and train_done:
            print("SKIPPING")
            print(model_name)
            continue
    model = load_model(name=model_type, path=root_dir+model_path, device=device)
    model.eval() 
    model.return_z = False
    if args.temperature_scale:
        model = temperature_scale_model(model, dataset, batch_size)
    model = model.to(device)
    model.device = device
        

    if not train_done:
        nll_value = eval_train_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['Train nll'] = nll_value

    if not in_done:
        ece_calc, mce_calc, acc, nll_value, brier_score, OOD_y_preds_logits, OOD_labels = eval_test_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['clean_accuracy'] = acc.to("cpu").numpy().tolist()
        results[model_name]['ECE'] = ece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['MCE'] = mce_calc.to("cpu").numpy().tolist()*100
        results[model_name]['nll'] = nll_value
        results[model_name]['brier'] = brier_score

    if not shift_done:
        ece_calc, mce_calc, acc, corruption_ece_dict, corruption_mce_dict = eval_shift_data(model, dataset=dataset, batch_size=batch_size, device=device, num_samples=args.num_samples)
        results[model_name]['SHIFT ECE'] = ece_calc.to("cpu").numpy().tolist()*100
        results[model_name]['SHIFT MCE'] = mce_calc.to("cpu").numpy().tolist()*100
        results[model_name]['SHIFT ACCURACY'] = acc.to("cpu").numpy().tolist()
        for key in corruption_ece_dict.keys():
            results[model_name]["SHIFT Intensity: " + str(key)] = {}
            results[model_name]["SHIFT Intensity: " + str(key)]['ECE'] = corruption_ece_dict[key]
            results[model_name]["SHIFT Intensity: " + str(key)]['MCE'] = corruption_mce_dict[key]

    if not ood_done:
        auroc_calc, fpr_at_95_tpr_calc = eval_ood_data(model, dataset=dataset, batch_size=batch_size, device=device, OOD_y_preds_logits=OOD_y_preds_logits, OOD_labels=OOD_labels, num_samples=args.num_samples)
        results[model_name]['OOD AUROC'] = auroc_calc
        results[model_name]['OOD FPR95'] = fpr_at_95_tpr_calc
    with open('./experiment_results/table_metrics/'+args.save_file_name, 'w') as fp:
        json.dump(results, fp)

    num_models += 1


if num_models > 1:
    output_file = args.save_file_name.replace('.', '_summary.')
    model_results = open('./experiment_results/table_metrics/'+args.save_file_name, 'r')

    metrics_data = {}
    for line_data in model_results.read().splitlines():
        data = json.loads(line_data)
        
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
        metrics_summary[metric] = {
            "average": np.mean(values),
            "SE": np.std(values)/np.sqrt(num_models)
        }


    with open('./experiment_results/table_metrics/'+output_file, 'w') as output:
        json.dump(metrics_summary, output, indent=4)

    print(f"Metrics summary saved to {'./experiment_results/table_metrics/'+output_file}")
    