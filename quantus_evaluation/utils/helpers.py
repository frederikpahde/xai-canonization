import logging

import numpy as np
import pickle
import os


logger = logging.getLogger(__name__)

def load_or_init_results(results_path, xai_methods):
    """Initializes empty results dictionary or loads existing one.

    Args:
        results_path (str): Path to existing results dict
        xai_methods (dictionary): dictionary, where keys are names of xai methods to be evaluated

    Returns:
        dictionary: initialized/loaded results dictionary
    """
    logger.info(f"Check whether path exists: {results_path}")
    # Prepare results dict
    if results_path and os.path.isfile(results_path):
        logger.info(f"Found past results! Loading ...")
        results = pickle.load(open(results_path, "rb"))
    else:
        logger.info(f"Nothing found! Initializing empty results dictionary ...")
        results = {}
        
    for method in xai_methods.keys():
        if method in results.keys():
            logger.info(f"Results exist for method {method}")
        else:
            results[method] = {}

    return results

def compute_aopc(scores_class):
    """Computes Area over Pertubuation Curve

    Args:
        scores_class (np.Array): array with class-specific results from pixel flipping experiment

    Returns:
        scalar: area over pertubation curve
    """
    # print("scores_class", scores_class.shape, scores_class)
    y0 = scores_class[:,0].reshape(-1,1)
    L = scores_class.shape[1]
    y0_expanded = np.broadcast_to(y0, scores_class.shape)
    aopc = (y0_expanded - scores_class).mean(axis=0).sum() / L
    return aopc
