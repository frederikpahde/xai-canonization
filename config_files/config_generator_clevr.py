import yaml
import os
import copy

USER_VARIABLES = {
    'device': 'cuda',  # device name for torch
    'dataset_path': '<DATASET_PATH>',  # dataset root path
    'model_path': '<MODEL_PATH>',
    'batch_size': 64,
    'num_batches_to_process': 50,
    'num_batches_to_process_pixel_flipping': 25
}

base_config = {
    'seed': 1,
    'input_size': 128,
    'dataset_name': 'clevr',
    'model_name': 'relation_network',

}

for key, val in USER_VARIABLES.items():
    base_config[key] = val

config_dir = "./"


def create_config(config, config_name):
    path = f"{config_dir}/clevr_xai"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/{config_name}.yaml", 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


base_config_name = 'clevr_xai'

for fn_pool_name in ('pos_sq_sum', 'max_norm'):
    for xai_method in [
        "excitation_backprop",
        'lrp_rn',
        'lrp_rn_conv_canonized',
        'lrp_rn_all_canonized'
    ]:
        for question_type in ['simple', 'complex']:
            base_config['question_type'] = question_type
            base_config['xai_method'] = xai_method
            base_config['fn_pool_name'] = fn_pool_name

            xai_method_str = xai_method.replace('.', '')
            for c in ('(', ')', ' ', '__'):
                xai_method_str = xai_method_str.replace(c, '_')

            config_name = f"{base_config_name}_{question_type}_{fn_pool_name}_{xai_method_str}"
            create_config(copy.deepcopy(base_config), config_name)
