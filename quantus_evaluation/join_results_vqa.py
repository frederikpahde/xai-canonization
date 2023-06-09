import argparse
import logging
import os
import glob
import numpy as np
import json
from evaluate_explainers import summarize_results

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)



def join_results(results_dir, pooling, question_type):
    """Joins results of multiple runs and stores them in a single directory. 
    All runs are expected to be subdirs in results_dir.
    A new subdir 'results_all' is created with aggregated results.

    Args:
        results_dir (str): directory with results from multiple runs.
    """
    subdirs = [p for p in glob.glob(f"{results_dir}/clevr_xai_{question_type}_*") if (os.path.isdir(p) and (not p.endswith("all")))]
    subdirs = [subdir for subdir in subdirs if pooling in subdir]

    results = dict()
    y_all = None
    class_name_by_index_dict = None
    
    for dir in subdirs:
        if not os.path.isfile(f"{dir}/y_all.npy"):
            print(f"Skiping dir {dir} ... (y_all missing)")
            continue
        if not os.path.isfile(f"{dir}/eval_results.json"):
            print(f"Skiping dir {dir} ... (eval_results missing)")
            continue
        if not os.path.isfile(f"{dir}/class_name_by_index_dict.json"):
            print(f"Skiping dir {dir} ... (class_name_by_index_dict missing)")
            continue

        print(f"Collecting results from {dir}")

        # Read Ys
        y_all_subdir = np.load(f"{dir}/y_all.npy")

        # Read Eval Results
        with open(f"{dir}/eval_results.json") as file:
            results_subdir = json.load(file)

        if y_all is None:
            y_all = y_all_subdir
            with open(f"{dir}/class_name_by_index_dict.json") as file:
                class_name_by_index_dict = json.load(file)

        for xai_method_name, results_method_subdir in results_subdir.items():

            if xai_method_name in results.keys():
                for metric, values in results_method_subdir.items():
                    if metric in results[xai_method_name].keys():
                        if isinstance(values, dict):
                            for key, item in values.items():
                                results[xai_method_name][metric][key] += item
                        else:        
                            results[xai_method_name][metric] += values
                    else:
                        results[xai_method_name][metric] = values

            else:
                results[xai_method_name] = results_method_subdir

    # Summarize concatenated results
    save_dir = f"{results_dir}/results_all_{question_type}_{pooling}"
    os.makedirs(save_dir, exist_ok=True)
    class_name_by_index_dict = {int(key): item for key, item in class_name_by_index_dict.items()}
    df = summarize_results(results, y_all, class_name_by_index_dict, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--pooling', type=str)
    parser.add_argument('--question_type', type=str)
    args = parser.parse_args()
    results_dir = args.results_dir
    question_type = args.question_type
    pooling = args.pooling
    join_results(results_dir, pooling, question_type)
