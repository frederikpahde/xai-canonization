import yaml
import os
import copy

USER_VARIABLES = {
    'device': 'cuda',  # device name for torch
    'dataset_path': '<IMAGENET_IMAGE_PATH>',
    'annotation_path': '<IMAGENET_ANNOTATION_PATH>',
    'save_dir': 'test_outputs/',  # output directory
    'batch_size': 32,
    'num_batches_to_process': 20,
    'num_batches_to_process_pixel_flipping': 10
}

base_config = {'device': 'cuda',
               'img_size': 224,
               'label_map_path': 'other_files/label-map-imagenet.json',
               'high_gammas' : [0, .1, .25, .5, 1, 10, 50],
               'feat_gammas' : [0, .1, .25, .5, 1, 10, 50]
              }

for key, val in USER_VARIABLES.items():
    base_config[key] = val

config_dir = "./"
word_net_ids = ["n01843383", "n02097474"]

def create_config(config, config_name):
    path = f"{config_dir}/grid_search"
    os.makedirs(path, exist_ok=True)
    
    with open(f"{path}/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

for low_gamma  in [0, .1, .25, .5, 1, 10, 50]:
    base_config['low_gammas'] = [low_gamma]
    for mid_gamma  in [0, .1, .25, .5, 1, 10, 50]:
        base_config['mid_gammas'] = [mid_gamma]
        for word_net_id in word_net_ids:
            base_config['word_net_ids'] = [word_net_id]
            for canonized in [0, 1]:
                base_config['canonized'] = canonized
                config_name = f"{word_net_id}_can{canonized}_{low_gamma}_{mid_gamma}"
                create_config(copy.deepcopy(base_config), config_name)