import quantus ## Core dumped if not imported first

import json
import logging
import os
import numpy as np
import torch
from zennit import composites as zcomp
from zennit import attribution as zattr
from zennit.core import Composite

from explainers import (
    # Default
    explainer_zennit_default,

    # VGG
    explainer_zennit_bn_canonized,

    # ResNet
    explainer_zennit_resnet,
    explainer_zennit_resnet_canonized,

    # EfficientNet
    explainer_zennit_efficientnet,
    explainer_zennit_efficientnet_canonized,

    # DenseNet
    explainer_zennit_densenet,
    explainer_zennit_densenet_canonized_seq_thresh,
    explainer_zennit_densenet_canonized_thresh_seq,
    explainer_zennit_densenet_untangled,

    #RelationNetwork
    explainer_zennit_rn,
    explainer_zennit_rn_all_canonized,
    explainer_zennit_rn_bn_only_canonized)

from metrics import (
                    # Robustness Metrics
                    init_avg_sensitivity,
                    init_max_sensitivity,

                    # Faithfulness Metrics
                     init_pixel_flipping,
                     init_faithfulness_correlation,
                     init_region_perturbation_blur,

                     # Randomization Metrics
                     init_random_logit,

                     # Localization Metrics
                     init_relevanve_rank_accuracy, 
                     init_relevanve_mass_accuracy,

                     #Complexity Metrics
                     init_sparseness,
                     init_complexity)

from utils.helpers_plots import plot_attributions, plot_attributions_clevr

from utils.attributors_vqa import GradientVQA, IntegratedGradientsVQA, SmoothGradVQA
from utils.composites import ExcitationBackpropBox
from utils.pooling_functions import pool_pos_sq_sum, pool_sum, pool_max_norm
from utils.helpers_data import load_dataset, get_data_loader
from utils.helpers_model import load_model
from utils.helpers import (load_or_init_results)
from evaluation import (compute_rankings, run_evaluation, run_evaluation_multilabel, run_evaluation_clevr_xai)
from utils.helpers_plots import (plot_attributions, plot_attributions_clevr,
                                 aggregate_and_plot_pixel_flipping_experiments,
                                 plot_spyder_graph)

import torch
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'

logger = logging.getLogger(__name__)


def summarize_results(results, y_all, class_name_by_index_dict, save_dir):
    """Summarizes evaluation results in DataFrames and creates figures (pixel flipping, spider)

    Args:
        results (dictionary): evaluation results
        y_all (np.array): class labels of samples
        class_name_by_index_dict (dictionary): maps class label to class name
        save_dir (str): directory where results are saved
    """
    # Create Pixel Flipping plot
    if "PixelFlipping (mean)" in results[list(results.keys())[0]].keys():
        results = aggregate_and_plot_pixel_flipping_experiments(results, y_all, 'PixelFlipping (mean)', 'Faithfulness (PF - m.)', class_name_by_index_dict, save_dir)
    if "PixelFlipping (black)" in results[list(results.keys())[0]].keys():
        results = aggregate_and_plot_pixel_flipping_experiments(results, y_all, 'PixelFlipping (black)', 'Faithfulness (PF - bl.)', class_name_by_index_dict, save_dir)
    if "PixelFlipping (blur)" in results[list(results.keys())[0]].keys():
        results = aggregate_and_plot_pixel_flipping_experiments(results, y_all, 'PixelFlipping (blur)', 'Faithfulness (PF - blur)', class_name_by_index_dict, save_dir)
    if "RegionPerturbation (mean)" in results[list(results.keys())[0]].keys():
        results = aggregate_and_plot_pixel_flipping_experiments(results, y_all, 'RegionPerturbation (mean)', 'Faithfulness (RP - mean)', class_name_by_index_dict, save_dir)
    if "RegionPerturbation (blur)" in results[list(results.keys())[0]].keys():
        results = aggregate_and_plot_pixel_flipping_experiments(results, y_all, 'RegionPerturbation (blur)', 'Faithfulness (RP - blur)', class_name_by_index_dict, save_dir)

    # Convert results to pandas DataFrame
    df, df_normalised_rank = compute_rankings(results)
    path_df = f"{save_dir}/results_agg.csv"
    path_df_rankings = f"{save_dir}/rankings.csv"
    df.to_csv(path_df)
    df_normalised_rank.to_csv(path_df_rankings)

    if len(results.keys()) > 2:
        # Create Spyder Graph
        plot_spyder_graph(df_normalised_rank, save_dir)

    return df

def get_pixel_values_by_dataset(dataset_name):
    mean_pixel = np.array([0.463, 0.460 , 0.453]) if dataset_name == 'clevr' else np.zeros(3)
    black_pixel = np.zeros(3) if dataset_name == 'clevr' else np.array([-2.118, -2.036, -1.804])
    return {'black': black_pixel, 'mean': mean_pixel}

def to_float(results):
    if isinstance(results, dict):
        return {key: to_float(r) for key, r in results.items()}
    else:
        return np.array(results).astype(float).tolist()

def get_metrics(dataset_name):
    """ Definition of XAI metrics for Quantus evaluation

    Args:
        dataset_name (str): name of dataset to be evaluated

    Returns:
        dictionary: evaluation metrics, where key is name and value is a tuple with metric initializer and list of *args
    """
    if dataset_name == 'clevr':
        num_classes = 28
    elif dataset_name == 'VOC':
        num_classes = 20
    elif 'MS' in dataset_name:
        num_classes = 80
    else:
        num_classes = 1000
    pixel_flip_steps = 127 if dataset_name == 'clevr' else 448
    img_size = 128 if dataset_name == 'clevr' else 224
    
    pixel_values = get_pixel_values_by_dataset(dataset_name)
    metrics =  {
            "Robustness (Avg Sens.)": (init_avg_sensitivity, []),
            "Robustness (Max Sens.)": (init_max_sensitivity, []),
            "Localisation (RRA)": (init_relevanve_rank_accuracy, []),
            "Localisation (RMA)": (init_relevanve_mass_accuracy, []),
            "Complexity (Sp.)": (init_sparseness, []),
            "Complexity (Comp.)": (init_complexity, []),
            "Randomisation (Logit)": (init_random_logit, [num_classes]),
            "Faithfulness (Corr. small)": (init_faithfulness_correlation, [int(img_size * img_size * 0.01), pixel_values['mean']]),
            "PixelFlipping (mean)": (init_pixel_flipping, [pixel_flip_steps, pixel_values['mean']]),
            "RegionPerturbation (blur)": (init_region_perturbation_blur, [])
           }

    return metrics

def get_explainers_by_model(model_name, dataset_name, device):
    """ Definition of XAI methods to be evaluated.

    Args:
        model_name (str): name of model architecture to be evaluated
        dataset_name (str): name of dataset to be evaluated
        device (str): name of device
    Returns:
        dictionary: explainers, where key is name and value is wrapper for explainer
    """
    if model_name == "relation_network":
        ## This is a VQA model
        attributors = {'gradient': GradientVQA,
                       'smoothgrad': SmoothGradVQA,
                       'integrated_gradients': IntegratedGradientsVQA}
    else:
        attributors = {'gradient': zattr.Gradient,
                       'smoothgrad': zattr.SmoothGrad,
                       'integrated_gradients': zattr.IntegratedGradients}

    pixel_values = get_pixel_values_by_dataset(dataset_name)
    baseline_fn_mean = lambda x: torch.from_numpy(pixel_values['mean'].reshape(-1, 1, 1)).float().to(device) * torch.ones_like(x)
    # Default explainers
    explainers_default = {
        # 'name': (attributor, attributor_kwargs, explain_fn, composite)
        'saliency':  (attributors['gradient'], {}, explainer_zennit_default, Composite),
        'integrated_gradients': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_default, Composite),
        'smoothgrad': (attributors['smoothgrad'], {}, explainer_zennit_default, Composite),
        
        'guided_backprop': (attributors['gradient'], {}, explainer_zennit_default, zcomp.GuidedBackprop)
    }

    if "vgg" in model_name:
        explainers_model_specific = {'saliency_vgg_canonized':  (attributors['gradient'], {}, explainer_zennit_bn_canonized, Composite),
                                     'integrated_gradients_vgg_canonized': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_bn_canonized, Composite),
                                     'smoothgrad_vgg_canonized': (attributors['smoothgrad'], {}, explainer_zennit_bn_canonized, Composite),
                                     'excitation_backprop_vgg_canonized': (attributors['gradient'], {}, explainer_zennit_bn_canonized, zcomp.ExcitationBackprop),
                                     'guided_backprop_vgg_canonized': (attributors['gradient'], {}, explainer_zennit_bn_canonized, zcomp.GuidedBackprop),
                                     'excitation_backprop': (attributors['gradient'], {}, explainer_zennit_default, zcomp.ExcitationBackprop),
                                     'lrp_ep_vgg': (attributors['gradient'], {}, explainer_zennit_default, zcomp.EpsilonPlus),
                                     'lrp_a2b1_vgg': (attributors['gradient'], {}, explainer_zennit_default, zcomp.EpsilonAlpha2Beta1),
                                     'lrp_ep_vgg_canonized': (attributors['gradient'], {}, explainer_zennit_bn_canonized, zcomp.EpsilonPlus),
                                     'lrp_a2b1_vgg_canonized': (attributors['gradient'], {}, explainer_zennit_bn_canonized, zcomp.EpsilonAlpha2Beta1),
                                     }
    elif "resnet" in model_name:
        explainers_model_specific = {'excitation_backprop': (attributors['gradient'], {}, explainer_zennit_resnet, zcomp.ExcitationBackprop),
                                     'saliency_resnet_canonized':  (attributors['gradient'], {}, explainer_zennit_resnet_canonized, Composite),
                                     'integrated_gradients_resnet_canonized': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_resnet_canonized, Composite),
                                     'smoothgrad_resnet_canonized': (attributors['smoothgrad'], {}, explainer_zennit_resnet_canonized, Composite),
                                     'excitation_backprop_resnet_canonized': (attributors['gradient'], {}, explainer_zennit_resnet_canonized, zcomp.ExcitationBackprop),
                                     'guided_backprop_resnet_canonized': (attributors['gradient'], {}, explainer_zennit_resnet_canonized, zcomp.GuidedBackprop),
                                     'lrp_ep_resnet': (attributors['gradient'], {}, explainer_zennit_resnet, zcomp.EpsilonPlus),
                                     'lrp_a2b1_resnet': (attributors['gradient'], {}, explainer_zennit_resnet, zcomp.EpsilonAlpha2Beta1),
                                     'lrp_ep_resnet_canonized': (attributors['gradient'], {}, explainer_zennit_resnet_canonized, zcomp.EpsilonPlus),
                                     'lrp_a2b1_resnet_canonized': (attributors['gradient'], {}, explainer_zennit_resnet_canonized, zcomp.EpsilonAlpha2Beta1)
                                     }

    elif "efficientnet" in model_name:
        explainers_model_specific = {'excitation_backprop': (attributors['gradient'], {}, explainer_zennit_efficientnet, zcomp.ExcitationBackprop),
                                     'saliency_efficientnet_canonized':  (attributors['gradient'], {}, explainer_zennit_bn_canonized, Composite),
                                     'integrated_gradients_efficientnet_canonized': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_bn_canonized, Composite),
                                     'smoothgrad_efficientnet_canonized': (attributors['smoothgrad'], {}, explainer_zennit_bn_canonized, Composite),
                                     'excitation_backprop_efficientnet_canonized': (attributors['gradient'], {}, explainer_zennit_bn_canonized, zcomp.ExcitationBackprop),
                                     'guided_backprop_efficientnet_canonized': (attributors['gradient'], {}, explainer_zennit_bn_canonized, zcomp.GuidedBackprop),
                                     'lrp_ep_efficientnet': (attributors['gradient'], {}, explainer_zennit_efficientnet, zcomp.EpsilonPlus),
                                     'lrp_a2b1_efficientnet': (attributors['gradient'], {}, explainer_zennit_efficientnet, zcomp.EpsilonAlpha2Beta1),
                                     'lrp_ep_efficientnet_canonized': (attributors['gradient'], {}, explainer_zennit_efficientnet_canonized, zcomp.EpsilonPlus),
                                     'lrp_a2b1_efficientnet_canonized': (attributors['gradient'], {}, explainer_zennit_efficientnet_canonized, zcomp.EpsilonAlpha2Beta1)
                                     }

    elif "densenet" in model_name:
        explainers_model_specific = {'saliency_densenet_canonized_seq_thresh':  (attributors['gradient'], {}, explainer_zennit_densenet_canonized_seq_thresh, Composite),
                                     'saliency_densenet_canonized_thresh_seq':  (attributors['gradient'], {}, explainer_zennit_densenet_canonized_thresh_seq, Composite),
                                     'integrated_gradients_densenet_canonized_seq_thresh': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_densenet_canonized_seq_thresh, Composite),
                                     'integrated_gradients_densenet_canonized_thresh_seq': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_densenet_canonized_thresh_seq, Composite),
                                     'smoothgrad_densenet_canonized_seq_thresh': (attributors['smoothgrad'], {}, explainer_zennit_densenet_canonized_seq_thresh, Composite),
                                     'smoothgrad_densenet_canonized_thresh_seq': (attributors['smoothgrad'], {}, explainer_zennit_densenet_canonized_thresh_seq, Composite),
                                     'guided_backprop_densenet_canonized_seq_thresh': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_seq_thresh, zcomp.GuidedBackprop),
                                     'guided_backprop_densenet_canonized_thresh_seq': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_thresh_seq, zcomp.GuidedBackprop),
                                     'excitation_backprop': (attributors['gradient'], {}, explainer_zennit_densenet, zcomp.ExcitationBackprop),
                                     'lrp_ep_densenet': (attributors['gradient'], {}, explainer_zennit_densenet, zcomp.EpsilonPlus),
                                     'lrp_a2b1_densenet': (attributors['gradient'], {}, explainer_zennit_densenet, zcomp.EpsilonAlpha2Beta1),
                                     'excitation_backprop_densenet_canonized_seq_thresh': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_seq_thresh, zcomp.ExcitationBackprop),
                                     'lrp_ep_densenet_canonized_seq_thresh': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_seq_thresh, zcomp.EpsilonPlus),
                                     'lrp_a2b1_densenet_canonized_seq_thresh': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_seq_thresh, zcomp.EpsilonAlpha2Beta1),
                                     'excitation_backprop_densenet_canonized_thresh_seq': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_thresh_seq, zcomp.ExcitationBackprop),
                                     'lrp_ep_densenet_canonized_thresh_seq': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_thresh_seq, zcomp.EpsilonPlus),
                                     'lrp_a2b1_densenet_canonized_thresh_seq': (attributors['gradient'], {}, explainer_zennit_densenet_canonized_thresh_seq, zcomp.EpsilonAlpha2Beta1),
                                     'excitation_backprop_densenet_canonized_untangled': (attributors['gradient'], {}, explainer_zennit_densenet_untangled, zcomp.ExcitationBackprop),
                                     'lrp_ep_densenet_canonized_untangled': (attributors['gradient'], {}, explainer_zennit_densenet_untangled, zcomp.EpsilonPlus),
                                     'lrp_a2b1_densenet_canonized_untangled': (attributors['gradient'], {}, explainer_zennit_densenet_untangled, zcomp.EpsilonAlpha2Beta1)
                                     }

    elif model_name == "relation_network":
        explainers_model_specific = {
            'excitation_backprop': (attributors['gradient'], {}, explainer_zennit_rn, zcomp.ExcitationBackprop),
            'saliency_rn_canonized': (attributors['gradient'], {}, explainer_zennit_rn_bn_only_canonized, Composite),
            'integrated_gradients_rn_canonized': (attributors['integrated_gradients'], {'baseline_fn': baseline_fn_mean}, explainer_zennit_rn_bn_only_canonized, Composite),
            'smoothgrad_rn_canonized': (attributors['smoothgrad'], {}, explainer_zennit_rn_bn_only_canonized, Composite),
            'excitation_backprop_rn_canonized': (attributors['gradient'], {}, explainer_zennit_rn_bn_only_canonized, zcomp.ExcitationBackprop),
            'guided_backprop_rn_canonized': (attributors['gradient'], {}, explainer_zennit_rn_bn_only_canonized, zcomp.GuidedBackprop),
            'lrp_rn': (attributors['gradient'], {}, explainer_zennit_rn, ExcitationBackpropBox),
            'lrp_rn_all_canonized': (attributors['gradient'], {}, explainer_zennit_rn_all_canonized, ExcitationBackpropBox)  
        }

    else:
        explainers_model_specific = {}

    return {**explainers_default, **explainers_model_specific}


def get_y_all(loader, num_batches_to_process_pixel_flipping, dataset_name):
    """Returns y-values for first num_batches_to_process_pixel_flipping batches.
    This is required for the visualization of the pixel flipping experiments.

    Args:
        loader (torch.utils.data.DataLoader): dataloader for evaluation
        num_batches_to_process_pixel_flipping (int): number of batches to be processed
        dataset_name (str): name of dataset to be evaluated

    Returns:
        np.array: list with labels
    """
    y_all = []
    for i, (batch) in enumerate(loader):
        if i >= num_batches_to_process_pixel_flipping:
            break
        if dataset_name == 'clevr':
            y_batch = batch['answer']
        else:
            y_batch = batch[2]
        y_all.append(y_batch.detach().cpu().numpy())
    if len(y_all) > 0:
        y_all = np.concatenate(y_all)
    return y_all

def create_sample_attributions(model, loader, xai_methods, device, xai_lib,
                               img_size, denormalizer, save_dir):
    """ Creates figure with attributions with all xai-methods

    Args:
        model (torch.nn.Module): model to be evaluated
        loader (torch.utils.data.DataLoader): dataloader for evaluation
        xai_methods (dictionary): dictionary with method name as key and dictionary with attributor, xai function and composite as value
        device (str): name of device (cuda/cpu)
        xai_lib (str): xai lib (e.g., zennit)
        img_size (int): width/height of images
        denormalizer: function to denormalize the data
        save_dir (str): directory where figure is saved
    """
    x_batch, _, y_batch = iter(loader).next()
    attributions = {key: explainer(model=model,
                                   inputs=x_batch.detach().cpu().numpy(),
                                   targets=y_batch.detach().cpu().numpy(),
                                   device=device,
                                   xai_lib=xai_lib,
                                   attributor=attributor,
                                   composite=composite,
                                   normalise=False,
                                   **{"img_size": img_size,
                                      "attributor_kwargs": attributor_kwargs}
                                   )
                    for key, (attributor, attributor_kwargs, explainer, composite) in xai_methods.items()}
    plot_attributions(x_batch, attributions, denormalizer, save_dir)

def create_sample_attributions_clevr(model, loader, xai_methods, device, xai_lib, 
                                     img_size, fn_pool, save_dir, vocabularies):
    """ Creates figure with attributions with all xai-methods for CLEVR dataset

    Args:
        model (torch.nn.Module): model to be evaluated
        loader (torch.utils.data.DataLoader): dataloader for evaluation
        xai_methods (dictionary): dictionary with method name as key and dictionary with attributor, xai function and composite as value
        device (str): name of device (cuda/cpu)
        xai_lib (str): xai lib (e.g., zennit)
        img_size (int): width/height of images
        fn_pool (Callable): pooling function
        save_dir (str): directory where figure is saved
        vocabularies (tuple): tuple with dictionaries for question/answer vocabularies
    """

    batch = iter(loader).next()
    y_batch = batch['answer']
    img_batch = batch['image']
    q_batch = batch['question']
    q_lengths_batch = batch['len_q']

    model = model.eval()
    model_out = model(img_batch[:, :, :127, :127].to(device), q_batch.to(device), q_lengths_batch).detach().cpu()
    pred_batch = model_out.argmax(1)

    attributions = {key: explainer(model=model,
                                   inputs=img_batch[:, :, :127, :127].detach().cpu().numpy(),
                                   targets=y_batch.detach().cpu().numpy(),
                                   device=device,
                                   xai_lib=xai_lib,
                                   attributor=attributor,
                                   composite=composite,
                                   normalise=False,
                                   **{"img_size":img_size,
                                      "question": q_batch.detach().cpu().numpy(),
                                      "q_length": q_lengths_batch,
                                      "is_vqa": True,
                                      "fn_pool": fn_pool,
                                      "attributor_kwargs": attributor_kwargs}
                                    )
                        for key, (attributor, attributor_kwargs, explainer, composite) in xai_methods.items()}

    vocab_q, inv_vocab_a = vocabularies[0], {i: word for word, i in vocabularies[1].items()}
    plot_attributions_clevr(batch, attributions, pred_batch, save_dir, vocab_q, inv_vocab_a)

def evaluate_explainers(
        model_name,
        dataset_name,
        dataset_path,
        img_size,
        batch_size,
        device,
        num_batches_to_process,
        num_batches_to_process_pixel_flipping,
        seed,
        save_dir,
        fn_pool_name,
        xai_method,
        annotation_path,
        label_map_path,
        word_net_ids,
        question_type,
        classes,
        mask_type,
        model_path
):

                        
    """Run quantus evaluation experiments.

    Args:
        model_name (str): name of model to be evaluated
        model_path (str): path to model weights to be evaluated
        dataset_name (str): name of dataset to be evaluated
        dataset_path (str): path to Imagenet samples
        img_size (int): size of images
        batch_size (int): batch_size
        device (str): name of device (cuda/cpu)
        num_batches_to_process (int): stops evaluation after num_batches_to_process batches
        num_batches_to_process_pixel_flipping (int): stops pixel flipping after num_batches_to_process_pixel_flipping batches
        seed (int): random seed
        save_dir (str): directory where results are stored
        fn_pool_name (str): name of pooling function
        xai_method (str): name of xai method to be evaluated (if None: use all XAI methods)

        # ImageNet specific
        annotation_path (str): path to annotations
        label_map_path (str): path to label map
        word_net_ids (list[str]): list of word net ids to be considered for evaluation
        
       # CLEVR-XAI specific
        question_type (str): name of question type (simple/comple)

       # MS Coco and VOC specific
        classes (list[int]): list of indices to be selected from the VOC or MS Coco datasets. Should be left None to reproduce paper results
        mask_type (str): bbox for bounding box, segm for segmentation masks to use with localisation metrics

    """

    torch.manual_seed(seed)
    np.seed = seed

    logger.info(f"Results will be stored in {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Load Data
    if dataset_name == "imagenet":
        dataset_kwargs = {'label_map_path': label_map_path,
                          'annotation_path': annotation_path,
                          'word_net_ids': word_net_ids}
    elif dataset_name == "clevr":
        dataset_kwargs = {'question_type': question_type}
    elif dataset_name == "VOC":
        if classes is None:
            dataset_kwargs = {'classes': range(20), 'mask_type': mask_type}
        else:
            dataset_kwargs = {'classes': classes, 'mask_type': mask_type}
    elif "MS" in dataset_name:
        if classes is None:
            dataset_kwargs = {'mask_type': mask_type, 'annotation_path': annotation_path}
        else:
            dataset_kwargs = {'classes': classes, 'mask_type': mask_type, 'annotation_path': annotation_path}
    else:
        dataset_kwargs = {}

    for name, attr in dataset_kwargs.items():
        assert attr is not None, f"{name} required in config for {dataset_name}"

    dataset = load_dataset(dataset_name, dataset_path, img_size, **dataset_kwargs)
    logger.info(f"Found {len(dataset)} samples.")
    loader = get_data_loader(dataset_name, dataset, batch_size)

    # Load Model
    model = load_model(dataset_name, model_name, model_path=model_path).to(device).eval()

    # Load explainers / metrics
    xai_methods = get_explainers_by_model(model_name, dataset_name, device)

    if xai_method is not None:
        # Only evaluate for specific XAI method
        xai_methods = {key: item for key, item in xai_methods.items() if key == xai_method}

    metrics = get_metrics(dataset_name)

    logger.info(f"Explainers to evaluate: {list(xai_methods.keys())}")
    logger.info(f"Using Metrics from categories: {list(metrics.keys())}")

    # Configure LRP Details
    xai_lib = "zennit"

    fn_pool_map = {'sum': pool_sum,
                   'pos_sq_sum': pool_pos_sq_sum,
                   'max_norm': pool_max_norm}

    fn_pool = fn_pool_map[fn_pool_name]
    
    class_name_by_index_dict = dataset.get_class_name_by_index_dict()

    # Plot a couple of sample attributions
    if dataset_name == 'clevr':
        create_sample_attributions_clevr(model, loader, xai_methods, device, xai_lib,
                                        img_size, fn_pool, save_dir, dataset.get_vocabularies())
    else:
        create_sample_attributions(model, loader, xai_methods, device, xai_lib,
                                   img_size, dataset.get_denormalizer(), save_dir)
    # Run Evaluations for all explainers
    results_path = None #f"{results_filename_stem_class}.pickle"
    results_init = load_or_init_results(results_path, xai_methods)

    if dataset_name == 'clevr':
        results = run_evaluation_clevr_xai(xai_methods, metrics, loader, model, num_batches_to_process, num_batches_to_process_pixel_flipping,
                                device, xai_lib, img_size, fn_pool, results_init)
    elif 'VOC' in dataset_name or 'MS' in dataset_name:
        results = run_evaluation_multilabel(xai_methods, metrics, loader, model, num_batches_to_process,
                                            num_batches_to_process_pixel_flipping,
                                            device, xai_lib, img_size, results_init)
    else:
        results = run_evaluation(xai_methods, metrics, loader, model, num_batches_to_process,
                                 num_batches_to_process_pixel_flipping,
                                 device, xai_lib, img_size, results_init)


    results = to_float(results)

    with open(f"{save_dir}/eval_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    ## Store labels for pixel flipping experiment
    y_all = get_y_all(loader, num_batches_to_process_pixel_flipping, dataset_name)
    y_all_path = f"{save_dir}/y_all.npy"
    np.save(y_all_path, y_all)
    
    with open(f"{save_dir}/class_name_by_index_dict.json", 'w', encoding='utf-8') as f:
        json.dump(class_name_by_index_dict, f, ensure_ascii=False, indent=4)

    if len(y_all) > 0:
        summarize_results(results, y_all, class_name_by_index_dict, save_dir)
