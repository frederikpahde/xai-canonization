import quantus ## Core dumped if not imported first
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import logging
import argparse
import yaml
import json
import os
from zennit.rules import Gamma, Pass
from zennit.composites import NameMapComposite
from zennit.attribution import Gradient
from utils.helpers_model import load_model
from utils.helpers_data import load_imagenet
from explainers import explainer_zennit_default, explainer_zennit_bn_canonized
from metrics import (
                     init_avg_sensitivity,
                     init_pixel_flipping,
                     init_region_perturbation_blur,
                     init_random_logit,
                     init_relevanve_rank_accuracy, 
                     init_relevanve_mass_accuracy,
                     init_sparseness)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def to_float(results):
    if isinstance(results, dict):
        return {key: to_float(r) for key, r in results.items()}
    else:
        return np.array(results).astype(float).tolist()

def get_metrics():
    metrics =  {
            "Robustness (Avg Sens.)": (init_avg_sensitivity, []),
            "Localisation (RRA)": (init_relevanve_rank_accuracy, []),
            "Localisation (RMA)": (init_relevanve_mass_accuracy, []),
            "Complexity (Sp.)": (init_sparseness, []),
            "Randomisation (Logit)": (init_random_logit, [1000]),
            "PixelFlipping (mean)": (init_pixel_flipping, [448, np.zeros(3)]),
            "RegionPerturbation (blur)": (init_region_perturbation_blur, [])
            }

    return metrics

def get_name_map(model_name, low_gamma, mid_gamma, high_gamma, feat_gamma):
    name_map_by_model_name = {
        'vgg16': [
            (['features.0', 'features.3', 'features.7', 'features.10'], Gamma(low_gamma)),                  # low-level convolutions
            (['features.14', 'features.17', 'features.20', 'features.24', 'features.27'], Gamma(mid_gamma)), # mid-level convolutions
            (['features.30', 'features.34', 'features.37', 'features.40'], Gamma(high_gamma)),              # high-level convolutions
            (['classifier.0', 'classifier.3', 'classifier.6'], Gamma(feat_gamma)),            # fully connected layers
            ([f"features.{i}" for i in [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]], Pass()), # RELUs
            ([f"classifier.{i}" for i in [1, 4]], Pass()), # RELUs
            ([f"features.{i}" for i in [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]], Pass()), # BNs
        ],
        'resnet18':[
            (['conv1', 'layer1.0.conv1', 'layer1.0.conv2', 'layer1.1.conv1', 'layer1.1.conv2'], Gamma(low_gamma)),
            (['layer2.0.conv1', 'layer2.0.conv2', 'layer2.0.downsample.0', 
              'layer2.1.conv1', 'layer2.1.conv2', 
              'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.downsample.0', 
              'layer3.1.conv1', 'layer3.1.conv2', 
              ], Gamma(mid_gamma)),
            (['layer4.0.conv1', 'layer4.0.conv2', 'layer4.0.downsample.0', 
              'layer4.1.conv1', 'layer4.1.conv2'], Gamma(high_gamma)),
            (['fc'], Gamma(feat_gamma)),
        ]
    }

    assert model_name in name_map_by_model_name.keys(), f"No key map defined for model {model_name}"

    return name_map_by_model_name[model_name]

def get_y_all(loader, num_batches_to_process_pixel_flipping):
    """Returns y-values for first num_batches_to_process_pixel_flipping batches.
    This is required for the visualization of the pixel flipping experiments.

    Args:
        loader (torch.utils.data.DataLoader): dataloader for evaluation
        num_batches_to_process_pixel_flipping (int): number of batches to be processed

    Returns:
        np.array: list with labels
    """
    y_all = []
    for i, (batch) in enumerate(loader):
        if i >= num_batches_to_process_pixel_flipping:
            break
        y_batch = batch[2]
        y_all.append(y_batch.detach().cpu().numpy())
    if len(y_all) > 0:
        y_all = np.concatenate(y_all)
    return y_all

def run_grid_search(model_name, low_gammas, mid_gammas, high_gammas, feat_gammas, canonized,
                    dataset_path, word_net_ids, img_size, annotation_path,
                    label_map_path, batch_size, num_batches_to_process, num_batches_to_process_pixel_flipping,
                    device, save_dir):

    model = load_model("imagenet", model_name).to(device).eval()
    dataset = load_imagenet(dataset_path=dataset_path, 
                            img_size=img_size, 
                            label_map_path=label_map_path,
                            word_net_ids=word_net_ids,
                            annotation_path=annotation_path)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    metrics = get_metrics()
    xai_lib = 'zennit'
    fn_explainer = explainer_zennit_bn_canonized if canonized else explainer_zennit_default 

    results = {}

    for low_gamma in low_gammas:
        for mid_gamma in mid_gammas:
            for high_gamma in high_gammas:
                for feat_gamma in feat_gammas:
                    composite = NameMapComposite
                    composite_kwargs = {'name_map':get_name_map(model_name, low_gamma, mid_gamma, high_gamma, feat_gamma)}
                    
                    param_name = f"can{canonized}_{low_gamma}_{mid_gamma}_{high_gamma}_{feat_gamma}"
                    results[param_name] = {}

                    for metric_name, (metric_init, metric_params) in metrics.items():
                        logger.info(f"Starting eval explainer '{param_name}' with metric: {metric_name}")
                        
                        results[param_name][metric_name] = []
                        max_batch = num_batches_to_process_pixel_flipping if (("PixelFlipping" in metric_name) or ("RegionPer" in metric_name)) else num_batches_to_process

                        ## Iterate through data
                        for i, (x_batch, s_batch, y_batch) in enumerate(tqdm.tqdm(loader, total=min(max_batch, len(loader)))):

                            if i >= max_batch:
                                # Early exit
                                break
                            metric_func = metric_init(*metric_params)

                            scores = metric_func(model=model,
                                                x_batch=x_batch.detach().cpu().numpy(),
                                                y_batch=y_batch.detach().cpu().numpy(),
                                                a_batch=None,
                                                s_batch=s_batch.detach().cpu().numpy(),
                                                **{"device": device,
                                                    "explain_func": fn_explainer,
                                                    "explain_func_kwargs": {"xai_lib": xai_lib,
                                                                            "composite": composite,
                                                                            "composite_kwargs": composite_kwargs,
                                                                            "attributor": Gradient,
                                                                            "img_size": img_size}
                                                    })

                            if isinstance(scores, dict):
                                if not isinstance(results[param_name][metric_name], dict):
                                    results[param_name][metric_name] = scores
                                else:
                                    results[param_name][metric_name] = {key: existing_scores + scores[key] for key, existing_scores in 
                                                                                results[param_name][metric_name].items()}
                            else:
                                results[param_name][metric_name] += scores

    results = to_float(results)
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/eval_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    ## Store labels for pixel flipping experiment
    y_all = get_y_all(loader, num_batches_to_process_pixel_flipping)
    y_all_path = f"{save_dir}/y_all.npy"
    np.save(y_all_path, y_all)


def start_experiment(config_file_path, model_name):

    with open(config_file_path, "r") as stream:
        try:
            experiment_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    save_dir = f"{experiment_config['save_dir']}/{model_name}/{os.path.basename(config_file_path)[:-5]}"

    run_grid_search(model_name=model_name,
                    low_gammas=experiment_config['low_gammas'], 
                    mid_gammas=experiment_config['mid_gammas'], 
                    high_gammas=experiment_config['high_gammas'], 
                    feat_gammas=experiment_config['feat_gammas'],
                    canonized=experiment_config['canonized'],
                    dataset_path=experiment_config['dataset_path'], 
                    word_net_ids=experiment_config['word_net_ids'], 
                    img_size=experiment_config['img_size'], 
                    annotation_path=experiment_config['annotation_path'],
                    label_map_path=experiment_config['label_map_path'], 
                    batch_size=experiment_config['batch_size'], 
                    num_batches_to_process=experiment_config['num_batches_to_process'], 
                    num_batches_to_process_pixel_flipping=experiment_config['num_batches_to_process_pixel_flipping'],
                    device=experiment_config['device'],
                    save_dir=save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    config_file_path = args.config_file
    model_name = args.model_name
    start_experiment(config_file_path, model_name)