import json
from datasets.imagenet import ImageNetBBox, transform as transform_imagenet, denormalizer as imagenet_denormalizer
from torch.utils.data import DataLoader
from datasets.clevr_xai import (ClevrXAIDataset, 
                                get_transform as get_transform_clevr, 
                                collate_samples_from_pixels)
from datasets.PascalVOC import VOCSingleClassSegmentation, VOCSingleClassBBox
from datasets.MSCoco import CocoEvaluation


def load_dataset(dataset_name, dataset_path, img_size, **dataset_kwargs):
    if dataset_name == "imagenet":
        return load_imagenet(img_size, dataset_path, **dataset_kwargs)
    elif dataset_name == "clevr":
        return load_clevr_xai(img_size, dataset_path, **dataset_kwargs)
    elif dataset_name == "VOC":
        return load_VOC([img_size, img_size], dataset_path, **dataset_kwargs)
    elif "MS" in dataset_name:
        return load_MS([img_size, img_size], dataset_path, **dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_VOC(mask_shape, dataset_path, **dataset_kwargs):
    if dataset_kwargs["mask_type"] == "bbox":
        return VOCSingleClassBBox(root=dataset_path, classes=dataset_kwargs['classes'], download=False,
                                  mask_shape=mask_shape, img_transform=transform_imagenet,
                                  denormalizer=imagenet_denormalizer, image_set="val")
    else:
        return VOCSingleClassSegmentation(root=dataset_path, classes=dataset_kwargs['classes'], download=False,
                                          mask_shape=mask_shape, img_transform=transform_imagenet,
                                          denormalizer=imagenet_denormalizer, image_set="val")


def load_MS(mask_shape, dataset_path, **dataset_kwargs):
    if 'classes' in dataset_kwargs.keys():
        classes=dataset_kwargs['classes']
    else:
        classes=None
    return CocoEvaluation(image_path=dataset_path, annotation_path=dataset_kwargs['annotation_path'],
                          classes=classes, img_transform=transform_imagenet, mask_type=dataset_kwargs['mask_type'],
                          mask_shape=mask_shape, denormalizer=imagenet_denormalizer)


def load_imagenet(img_size, dataset_path, **dataset_kwargs):

    with open(dataset_kwargs['label_map_path']) as label_map_file:
        label_map = json.load(label_map_file)

    return ImageNetBBox(dataset_kwargs['annotation_path'], 
                        dataset_path, 
                        label_map, 
                        dataset_kwargs['word_net_ids'], 
                        transform_imagenet, 
                        img_size)

def load_clevr_xai(img_size, dataset_path, **dataset_kwargs):

    path_vocab_q = f"{dataset_path}/vocab_q.json"
    path_vocab_a = f"{dataset_path}/vocab_a.json"
    
    with open(path_vocab_q, 'r') as file:
        vocab_q = json.load(file)
    with open(path_vocab_a, 'r') as file:
        vocab_a = json.load(file)

    vocabularies = (vocab_q, vocab_a)

    pred_path = f"{dataset_path}/correct_labels_{dataset_kwargs['question_type']}.npy"
    dataset = ClevrXAIDataset(dataset_path, 
                              img_size, 
                              vocabularies, 
                              get_transform_clevr(img_size), 
                              correct_only=True, 
                              pred_path=pred_path,
                              **dataset_kwargs)

    return dataset

def get_data_loader(dataset_name, dataset, batch_size):
    if dataset_name == 'clevr':
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_samples_from_pixels)
    elif dataset_name in ['imagenet', 'VOC', 'MS']:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


    return loader