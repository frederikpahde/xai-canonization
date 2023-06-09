from xml.etree import ElementTree
import cv2
import xmltodict
import torch
import numpy as np
import collections
import os
import glob
from torchvision import transforms
from pathlib import Path
from PIL import Image

mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

normalizer = transforms.Normalize(mean, std)

denormalizer = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=1 / np.array(std)),
                                   transforms.Normalize(mean=-np.array(mean),
                                                        std=[1., 1., 1.])])


transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                normalizer])

def load_sample_data(img_path, annotations_dir, label_map):

    annotation_paths = glob.glob(os.path.join(annotations_dir, '*.xml'))

    imgs, masks, labels = [], [], []
    for path in annotation_paths:
        wnid = Path(path).stem.split('_')[0]
        labels.append(label_map[wnid]['label'])
        
        sample_path = os.path.join(img_path, f"{Path(path).stem}.JPEG")
        imgs.append(transform(Image.open(sample_path).convert('RGB')))
        
        masks.append(load_binary_mask(path, wnid))

    return torch.stack(imgs), torch.stack(masks), torch.Tensor(labels).long()

def get_label_name(label_map, label):
    for _, label_details in label_map.items():
        if label_details['label'] == label:
            return label_details['name']
    raise ValueError(f"Label {label} not found.")

def load_binary_mask(path, word_net_id, img_size=224):
    binary_mask = {}

    # Parse annotations.
    tree = ElementTree.parse(path)
    xml_data = tree.getroot()
    xmlstr = ElementTree.tostring(xml_data, encoding="utf-8", method="xml")
    annotation = dict(xmltodict.parse(xmlstr))['annotation']

    width = int(annotation["size"]["width"])
    height = int(annotation["size"]["height"])

    # Iterate objects.
    objects = annotation["object"]
    if type(objects) != list:
        mask = np.zeros((height, width), dtype=int)
        mask[int(objects['bndbox']['ymin']):int(objects['bndbox']['ymax']), 
            int(objects['bndbox']['xmin']):int(objects['bndbox']['xmax'])] = 1
        binary_mask[objects['name']] = mask

    else:
        for object in annotation['object']:
            if type(object) in (collections.OrderedDict, dict):
                if object['name'] in binary_mask.keys():
                    mask = binary_mask[object['name']]
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)

                mask[int(object['bndbox']['ymin']):int(object['bndbox']['ymax']),
                    int(object['bndbox']['xmin']):int(object['bndbox']['xmax'])] = 1

                binary_mask[object['name']] = mask

    # Preprocess binary masks to fit shape of image data.
    for key in binary_mask.keys():
        binary_mask[key] = cv2.resize(binary_mask[key],
                                    (img_size, img_size),
                                    interpolation=cv2.INTER_NEAREST).astype(np.int)[:, :, np.newaxis]
    mask = torch.tensor(np.array([binary_mask[word_net_id][:, :, 0]]))

    return mask