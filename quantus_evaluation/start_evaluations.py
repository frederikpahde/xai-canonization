import argparse
import logging
import os
import yaml

from evaluate_explainers import evaluate_explainers

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def start_evaluation(config_file_path):
    """Start quantus evaluation

    Args:
        config_file_path (str): Path to config file for evaluation experiment
    """
    with open(config_file_path, "r") as stream:
        try:
            experiment_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    save_dir = f"{experiment_config['save_dir']}/{os.path.basename(config_file_path)[:-5]}"

    evaluate_explainers(
        model_name=experiment_config['model_name'],
        model_path=experiment_config.get('model_path', None),
        dataset_name=experiment_config['dataset_name'],
        dataset_path=experiment_config['dataset_path'], 
        
        img_size=experiment_config['input_size'],
        batch_size=experiment_config.get('batch_size', 8), 
        device=experiment_config.get('device', 'cuda'), 
        num_batches_to_process=experiment_config.get('num_batches_to_process', 10), 
        num_batches_to_process_pixel_flipping=experiment_config.get('num_batches_to_process_pixel_flipping', 5), 
        seed=experiment_config.get('seed', 0), 
        save_dir=save_dir,
        fn_pool_name=experiment_config.get('fn_pool_name', 'sum'),
        xai_method=experiment_config.get('xai_method', None),
        
        ## Imagenet-specific
        annotation_path=experiment_config.get('annotation_path', None),
        label_map_path=experiment_config.get('label_map_path', None),
        word_net_ids=experiment_config.get('word_net_ids', None),

        ## CLEVR-XAI-specific
        question_type=experiment_config.get('question_type', None),

        ## MS Coco and Pascal VOC-specific
        classes=experiment_config.get('classes', None),
        mask_type=experiment_config.get('mask_type', None)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file
    start_evaluation(config_file)
