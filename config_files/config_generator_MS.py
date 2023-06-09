import yaml
import os
import copy
import numpy as np
import json

USER_VARIABLES = {
    'device': 'cuda',  # device name for torch
    'dataset_path': '<COCO_IMAGE_PATH>',
    'annotation_path': '<COCO_ANNOTATION_PATH>',
    'model_path': '<MODEL_PATH>',
    'save_dir': 'test_outputs/',  # output directory
    'mask_type': 'segm',  # Type of localisation: 'segm' for segmentation masks and 'bbox' for bounding boxes
    'batch_size': 32,
    'num_batches_to_process': 20,
    'num_batches_to_process_pixel_flipping': 10

}

base_config = {'seed': 1,
               'dataset_name': 'MS',
               'input_size': 224,
               }
for key, val in USER_VARIABLES.items():
    base_config[key] = val

config_dir = "./"
NUM_CLASSES = 10
rng = np.random.default_rng(seed=42)
class_list = [int(i) for i in rng.permutation(80)[0:10]]


def create_config(config, config_name):
    path = f"{config_dir}/MS"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


xai_methods = ["excitation_backprop"]

model_specific_xai_methods = {
    "vgg": [
        'excitation_backprop_vgg_canonized',
        'lrp_ep_vgg',
        'lrp_a2b1_vgg',
        'lrp_ep_vgg_canonized',
        'lrp_a2b1_vgg_canonized'
    ],
    "resnet": ['excitation_backprop_resnet_canonized',
               'lrp_ep_resnet',
               'lrp_a2b1_resnet',
               'lrp_ep_resnet_canonized',
               'lrp_a2b1_resnet_canonized'
               ],
    "efficientnet": ['excitation_backprop_efficientnet_canonized',
                     'lrp_ep_efficientnet',
                     'lrp_a2b1_efficientnet',
                     'lrp_ep_efficientnet_canonized',
                     'lrp_a2b1_efficientnet_canonized'],
    "densenet": ['excitation_backprop_densenet_canonized_seq_thresh',
                 'excitation_backprop_densenet_canonized_thresh_seq',
                 'excitation_backprop_densenet_canonized_untangled',
                 'lrp_ep_densenet',
                 'lrp_a2b1_densenet',
                 'lrp_ep_densenet_canonized_seq_thresh',
                 'lrp_a2b1_densenet_canonized_seq_thresh',
                 'lrp_ep_densenet_canonized_thresh_seq',
                 'lrp_a2b1_densenet_canonized_thresh_seq',
                 'lrp_ep_densenet_canonized_untangled',
                 'lrp_a2b1_densenet_canonized_untangled',
                 ],

}

for model_name in [
    'efficientnet_b0',
    'efficientnet_b4',
    'densenet_121',
    'densenet_161',
    'densenet_169',
    'densenet_201',
    'resnet18',
    'resnet50',
    'vgg16'
]:

    for model_type in list(model_specific_xai_methods.keys()):
        if model_type in model_name:
            xai_methods_ms = model_specific_xai_methods[model_type]

    for xai_method in xai_methods + xai_methods_ms:
        base_config['xai_method'] = xai_method
        for cls in class_list:
            base_config['model_name'] = model_name
            xai_method_str = 'all' if xai_method is None else xai_method.replace('.', '')
            for c in ('(', ')', ' ', '__'):
                xai_method_str = xai_method_str.replace(c, '_')
            base_config['classes'] = [cls]
            config_name = f"{cls}_{model_name.lower()}_{xai_method_str}"
            create_config(copy.deepcopy(base_config), config_name)
