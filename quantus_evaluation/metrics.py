import quantus
import numpy as np
from typing import Optional, Sequence

def normalize_by_second_moment(a: np.ndarray,
                              normalise_axes: Optional[Sequence[int]] = None) -> np.ndarray:
    if normalise_axes is None:
        normalise_axes = list(range(np.ndim(a)))
    normalise_axes = tuple(normalise_axes)

    second_moment = np.mean(a**2, axis=normalise_axes, keepdims=True)
    second_moment_sqrt = second_moment ** .5
    return a / second_moment_sqrt

def init_avg_sensitivity():
    return quantus.AvgSensitivity(**{"nr_samples": 10,
                                     "perturb_func_kwargs": {"perturb_std": 0.2},
                                     "norm_numerator": quantus.norm_func.fro_norm,
                                     "norm_denominator": quantus.norm_func.fro_norm,
                                     "perturb_func": quantus.perturb_func.uniform_noise,
                                     "similarity_func": quantus.similarity_func.difference,
                                     "disable_warnings": True,
                                     "normalise_func": normalize_by_second_moment, 
                                     "normalise": True})

def init_max_sensitivity():
    return quantus.MaxSensitivity(**{"nr_samples": 10,
                                     "perturb_func_kwargs": {"perturb_std": 0.2},
                                     "norm_numerator": quantus.norm_func.fro_norm,
                                     "norm_denominator": quantus.norm_func.fro_norm,
                                     "perturb_func": quantus.perturb_func.uniform_noise,
                                     "similarity_func": quantus.similarity_func.difference,
                                     "disable_warnings": True,
                                     "normalise": True,
                                     "normalise_func": normalize_by_second_moment})


def init_relevanve_rank_accuracy(): 
    return quantus.RelevanceRankAccuracy(**{"abs": True,  
                                            "normalise": True,
                                            "normalise_func": normalize_by_second_moment, 
                                            "disable_warnings": True})

def init_relevanve_mass_accuracy(): 
    return quantus.RelevanceMassAccuracy(**{"abs": True,  
                                            "normalise": True,
                                            "normalise_func": normalize_by_second_moment, 
                                            "disable_warnings": True})

def init_sparseness(): 
    return quantus.Sparseness(**{"abs": True,  
                                 "normalise": True,
                                 "normalise_func": normalize_by_second_moment, 
                                 "disable_warnings": True})

def init_complexity(): 
    return quantus.Complexity(**{"abs": True,  
                                 "normalise": True,
                                 "normalise_func": normalize_by_second_moment, 
                                 "disable_warnings": True})

def init_random_logit(num_classes): 
    return quantus.RandomLogit(**{"abs": False,  
                                  "normalise": True,
                                  "normalise_func": normalize_by_second_moment, 
                                  "num_classes": num_classes,
                                  "similarity_func": quantus.similarity_func.ssim,
                                  "disable_warnings": True})

def init_pixel_flipping(features_in_step, perturb_baseline): 
    return quantus.PixelFlipping(**{"features_in_step": features_in_step, 
                                    "perturb_baseline": perturb_baseline,
                                    "perturb_func": quantus.perturb_func.baseline_replacement_by_indices,
                                    "disable_warnings": True})

def init_pixel_flipping_blur(features_in_step): 
    return quantus.PixelFlipping(**{"features_in_step": features_in_step, 
                                    "perturb_func": quantus.perturb_func.baseline_replacement_by_blur,
                                    "disable_warnings": True})

def init_faithfulness_correlation(subset_size, perturb_baseline):
    return quantus.FaithfulnessCorrelation(**{"perturb_baseline": perturb_baseline,
                                              "nr_runs": 100,  
                                              "subset_size": subset_size,
                                              "similarity_func": quantus.similarity_func.correlation_pearson,
                                              "abs": False,
                                              "normalise": True,
                                              "normalise_func": normalize_by_second_moment, 
                                              "return_aggregate": False,
                                              "disable_warnings": True})

def init_region_perturbation_blur():
    return quantus.RegionPerturbation(**{"patch_size": 14,
                                         "regions_evaluation": 50,
                                         "perturb_func": quantus.perturb_func.baseline_replacement_by_blur,
                                         "normalise": True,
                                         "disable_warnings": True})                                         