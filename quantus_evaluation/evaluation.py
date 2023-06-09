import gc
import logging
import pandas as pd
import numpy as np
import tqdm
import torch

logger = logging.getLogger(__name__)

def compute_rankings(results):
    """ Computes rankings of XAI methods per metric given results dictionary.

    Args:
        results (dictionary): result dictionary

    Returns:
        (df.DataFrame, df.DataFrame): (DataFrame with XAI method as index, metric as column and scalar as value
                                       DataFrame with XAI method as index, metric as column and rank as value)
    """
    results_agg = {}

    for method_name, results_method in results.items():
        results_agg[method_name] = {}
        for metric, values in results_method.items():
            if ("PixelFlipping" not in metric) and ("(Param.)" not in metric) and ("RegionPerturbation" not in metric):
                results_agg[method_name][metric] = np.nanmean(values)

    desc_sort_cols = ['Robustness', 'Randomisation']

    def getAscSortInds(cols):
        inds = np.array([False if col in desc_sort_cols else True for col in cols])
        return inds

    df = pd.DataFrame.from_dict(results_agg).T.abs().sort_index(axis=1)
    df_normalised = df.loc[:, getAscSortInds(df.columns)].apply(lambda x: x / x.max())
    for col in desc_sort_cols:
        if col in df.columns:
            df_normalised[col] = df[col].min()/df[col].values  
    df_normalised_rank = df_normalised.rank()

    return df, df_normalised_rank

def run_evaluation(xai_methods, metrics, loader, model, num_batches_to_process, num_batches_to_process_pixel_flipping,
                   device, xai_lib, img_size, results):
    """ Runs Quantus evaluation

    Args:
        xai_methods (dictionary): dictionary with method name as key and dictionary with attributor, xai function and composite as value
        metrics (dictionary): metric name as key, value is tuple with metric init function and list of params to pass
        loader (torch.utils.data.DataLoader): Dataloader used for evaluation
        model (torch.nn.Module): model to be evaluated
        num_batches_to_process (int): stops evaluation after num_batches_to_process batches
        num_batches_to_process_pixel_flipping (int): stops pixel flipping after num_batches_to_process_pixel_flipping batches
        device (str): name of device (cuda/cpu)
        xai_lib (str): xai lib (e.g., zennit)
        img_size (int): width/height of images
        results (dictionary): initialized results dict

    Returns:
        dictionary: results dictionary
    """
    for xai_method_name, (attributor, attributor_kwargs, xai_method_func, composite) in xai_methods.items():
        # Iterate over XAI metrics
        for metric_name, (metric_init, metric_params) in metrics.items():   
            logger.info(f"Starting eval explainer '{xai_method_name}' with metric: {metric_name}")
            results[xai_method_name][metric_name] = []
            # Iterate over batches in dataloader
            max_batch = num_batches_to_process_pixel_flipping if (("PixelFlipping" in metric_name) or ("RegionPerturbation" in metric_name)) else num_batches_to_process
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
                                        "explain_func": xai_method_func,
                                        "explain_func_kwargs": {"xai_lib": xai_lib,
                                                                "composite": composite,
                                                                "attributor": attributor,
                                                                "attributor_kwargs": attributor_kwargs,
                                                                "img_size": img_size}
                                         })

                if isinstance(scores, dict):
                    if not isinstance(results[xai_method_name][metric_name], dict):
                        results[xai_method_name][metric_name] = scores
                    else:
                        results[xai_method_name][metric_name] = {key: existing_scores + scores[key] for key, existing_scores in 
                                                                    results[xai_method_name][metric_name].items()}
                else:
                    results[xai_method_name][metric_name] += scores

                gc.collect()
                torch.cuda.empty_cache()
    return results

def run_evaluation_multilabel(xai_methods, metrics, loader, model, num_batches_to_process, num_batches_to_process_pixel_flipping,
                   device, xai_lib, img_size, results):
    """ Runs Quantus evaluation

    Args:
        xai_methods (dictionary): dictionary with method name as key and (callable) xai function as value
        metrics (dictionary): metric name as key, value is tuple with metric init function and list of params to pass
        loader (torch.utils.data.DataLoader): Dataloader used for evaluation
        model (torch.nn.Module): model to be evaluated
        num_batches_to_process (int): stops evaluation after num_batches_to_process batches
        num_batches_to_process_pixel_flipping (int): stops pixel flipping after num_batches_to_process_pixel_flipping batches
        device (str): name of device (cuda/cpu)
        xai_lib (str): xai lib (e.g., zennit)
        composite (zennit.core.Composite): composite used for zennit explainers
        attributor (zennit.attribution.Attributor): attributor used for zennit explainers
        img_size (int): width/height of images
        results (dictionary): initialized results dict

    Returns:
        dictionary: results dictionary
    """
    for xai_method_name, (attributor, attributor_kwargs, xai_method_func, composite) in xai_methods.items():
        # Iterate over XAI metrics
        for metric_name, (metric_init, metric_params) in metrics.items():
            logger.info(f"Starting eval explainer '{xai_method_name}' with metric: {metric_name}")
            results[xai_method_name][metric_name] = []
            # Iterate over batches in dataloader
            max_batch = num_batches_to_process_pixel_flipping if (("PixelFlipping" in metric_name) or ("Reg. Per." in metric_name)) else num_batches_to_process
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
                                        "softmax": False,
                                        "explain_func": xai_method_func,
                                        "explain_func_kwargs": {"xai_lib": xai_lib,
                                                                "composite": composite,
                                                                "attributor": attributor,
                                                                "attributor_kwargs": attributor_kwargs,
                                                                "img_size": img_size}
                                         })

                if isinstance(scores, dict):
                    if not isinstance(results[xai_method_name][metric_name], dict):
                        results[xai_method_name][metric_name] = scores
                    else:
                        results[xai_method_name][metric_name] = {key: existing_scores + scores[key] for key, existing_scores in
                                                                    results[xai_method_name][metric_name].items()}
                else:
                    results[xai_method_name][metric_name] += scores

                gc.collect()
                torch.cuda.empty_cache()
    return results


def run_evaluation_clevr_xai(xai_methods, metrics, loader, model, num_batches_to_process, num_batches_to_process_pixel_flipping,
                   device, xai_lib, img_size, fn_pool, results):
    """ Runs Quantus evaluation

    Args:
        xai_methods (dictionary): dictionary with method name as key and dictionary with attributor, xai function and composite as value
        metrics (dictionary): metric name as key, value is tuple with metric init function and list of params to pass
        loader (torch.utils.data.DataLoader): Dataloader used for evaluation
        model (torch.nn.Module): model to be evaluated
        num_batches_to_process (int): stops evaluation after num_batches_to_process batches
        num_batches_to_process_pixel_flipping (int): stops pixel flipping after num_batches_to_process_pixel_flipping batches
        device (str): name of device (cuda/cpu)
        xai_lib (str): xai lib (e.g., zennit)
        img_size (int): width/height of images
        results (dictionary): initialized results dict

    Returns:
        dictionary: results dictionary
    """
    for xai_method_name, (attributor, attributor_kwargs, xai_method_func, composite) in xai_methods.items():
        # Iterate over XAI metrics
        for metric_name, (metric_init, metric_params) in metrics.items():   
            if metric_name == "Randomisation (Param.)":
                # not supported
                continue
            logger.info(f"Starting eval explainer '{xai_method_name}' with metric: {metric_name}")
            results[xai_method_name][metric_name] = []
            # Iterate over batches in dataloader
            max_batch = num_batches_to_process_pixel_flipping if (("PixelFlipping" in metric_name) or ("RegionPerturbation" in metric_name)) else num_batches_to_process
            for i, (batch) in enumerate(tqdm.tqdm(loader, total=min(max_batch, len(loader)))):
                if i >= max_batch:
                    # Early exit
                    break
                y_batch = batch['answer']
                img_batch = batch['image']
                q_batch = batch['question']
                s_batch = batch['gt_single']
                q_lengths_batch = batch['len_q']

                model.coord_tensor = None

                metric_func = metric_init(*metric_params)
                scores = metric_func(model=model,
                                     x_batch=img_batch[:, :, :127, :127].detach().cpu().numpy(),
                                     y_batch=y_batch.detach().cpu().numpy(),
                                     a_batch=None,
                                     s_batch=s_batch[:, :, :127, :127].detach().cpu().numpy(),
                                     **{"device": device,
                                        "explain_func": xai_method_func,
                                        "explain_func_kwargs": {
                                                "xai_lib": xai_lib,
                                                "composite": composite,
                                                "attributor": attributor,
                                                "attributor_kwargs": attributor_kwargs,
                                                "img_size": img_size,
                                                "channel_first": True,
                                                "is_vqa": True,
                                                "question": q_batch.detach().cpu().numpy(),
                                                "q_length": q_lengths_batch,
                                                "fn_pool": fn_pool}
                                         })

                if isinstance(scores, dict):
                    if not isinstance(results[xai_method_name][metric_name], dict):
                        results[xai_method_name][metric_name] = scores
                    else:
                        results[xai_method_name][metric_name] = {key: existing_scores + scores[key] for key, existing_scores in 
                                                                    results[xai_method_name][metric_name].items()}
                else:
                    results[xai_method_name][metric_name] += scores

                gc.collect()
                torch.cuda.empty_cache()
    return results
