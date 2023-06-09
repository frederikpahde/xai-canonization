import collections
import glob
import logging
import os
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np
import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


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

class ImageNetBBox(Dataset):

    def __init__(self, annotation_path, image_path, label_map, classes, transforms, img_size=224):


        # Stores the arguments for later use
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.label_map = label_map
        self.classes = classes
        self.transforms = transforms
        self.img_size = img_size

        # Validates the arguments, by checking if the input path and the label map file exist
        assert os.path.isdir(self.image_path), "The image path does not exist or is not a directory."
        assert os.path.isdir(self.annotation_path), "The annotation path does not exist or is not a directory."

        self.samples = []
        self.bounding_boxes = []
        for word_net_id in self.classes:
            assert word_net_id  in self.label_map, f"The class {word_net_id} is invalid."
            for annotation_path in glob.glob(os.path.join(self.annotation_path, word_net_id, '*.xml')):
                sample_path = os.path.join(self.image_path, word_net_id, f"{Path(annotation_path).stem}.JPEG")
                if os.path.isfile(sample_path):
                    self.samples.append((sample_path, annotation_path, word_net_id))
        logger.info(f"Loaded {len(self.samples)} samples.")

    def load_binary_mask(self, path, word_net_id):
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
                                        (self.img_size, self.img_size),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int)[:, :, np.newaxis]
        mask = torch.tensor(np.array([binary_mask[word_net_id][:, :, 0]]))

        return mask

    def get_denormalizer(self):
        return denormalizer

    def get_class_name_by_index_dict(self):
        return {item['label']: item['name'] for _, item in self.label_map.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, annotation_path, word_net_id = self.samples[index]
        image = Image.open(path).convert('RGB')
        label = self.label_map[word_net_id]['label']
        image = self.transforms(image)
        mask = self.load_binary_mask(annotation_path, word_net_id)
        return image, mask, label
