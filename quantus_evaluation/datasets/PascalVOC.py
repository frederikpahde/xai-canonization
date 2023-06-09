import torchvision
import torch
import numpy as np
import os, logging
from xml.etree import ElementTree
from PIL import Image
import matplotlib.pyplot as plt
from .imagenet import transform as imagenet_transform
from .imagenet import denormalizer as imagenet_denormalizer
from torchvision.transforms.functional import to_pil_image, pil_to_tensor, resize

logger = logging.getLogger(__name__)
ALL_VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


# label_dict should be a dictionary with keys given in classes, and values corresponding to output tensor indices
# returns img, mask, label where label is integer, mask is a binary mask of bounding boxes
class VOCSingleClassBBox(torchvision.datasets.VOCDetection):
    all_classes = ALL_VOC_CLASSES

    def __init__(self, root, classes=None, label_dict={}, img_transform=lambda x:x,
                 denormalizer=lambda x:x,
                 download=True, include_difficult=False, image_set="val", mask_shape=[224, 224]):
        super().__init__(root=root, download=download, image_set=image_set)
        if classes is not None:
            self.classes=[self.all_classes[c] for c in classes]
        else:
            self.classes = self.all_classes
        self.img_transform = img_transform
        self.mask_shape = mask_shape
        if len(label_dict.keys()) == 0:
            self.label_dict = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            self.label_dict = label_dict
        self.denormalizer = denormalizer
        self.samples = []
        for i in range(len(self.annotations)):
            ann = self.parse_voc_xml(ElementTree.parse(self.annotations[i]).getroot())
            addedClasses = []
            for o in ann["annotation"]["object"]:
                if o["name"] in self.classes and o["name"] not in addedClasses:
                    if o["difficult"] == "0" or include_difficult:
                        self.samples.append((i, o["name"]))
                        addedClasses.append(o["name"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        index, cls = self.samples[item]
        img, ann = super().__getitem__(index)
        mask = np.zeros(img.size, dtype=np.int32)
        objects = ann["annotation"]["object"]
        for o in objects:
            if o["name"] == cls:
                mask[int(o['bndbox']['ymin']):int(o['bndbox']['ymax']),
                int(o['bndbox']['xmin']):int(o['bndbox']['xmax'])] = 1
        # return self.img_transform(img), np.expand_dims(mask, axis=0), self.label_dict[cls]
        if self.mask_shape is not None:
            mask = pil_to_tensor(resize(to_pil_image(mask * 255), self.mask_shape))
            mask[mask > 0] = 1
            return self.img_transform(img), mask, self.label_dict[cls]
        else:
            return self.img_transform(img), np.expand_dims(mask, axis=0), self.label_dict[cls]

    def get_denormalizer(self):
        return self.denormalizer

    def get_class_name_by_index_dict(self):
        return {i: self.all_classes[i] for i in range(20)}


# Returns img, mask, label where label is an int. mask is a binary segmantation mask
class VOCSingleClassSegmentation(torchvision.datasets.VOCSegmentation):
    all_classes=range(20)
    def __init__(self, root, classes=None, mask_shape=None, img_transform=lambda x:x, download=False,
                 denormalizer=lambda x:x, image_set="val"):
        super().__init__(root=root, download=download, image_set=image_set)
        if classes is not None:
            self.classes = classes
        else:
            self.classes = range(20)
        self.mask_shape = mask_shape
        self.img_transform = img_transform
        self.denormalizer = denormalizer
        self.samples = []

        for i in range(len(self.masks)):
            targets = np.unique(np.array(Image.open(self.masks[i]), dtype=int))
            addedClasses = []
            for c in self.classes:
                if c + 1 in targets and c not in addedClasses:
                    self.samples.append((i, c))
                    addedClasses.append(c)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        index, cls = self.samples[item]
        img, mask = super().__getitem__(index)
        mask = np.array(mask, dtype=np.int32)
        mask[mask == 255] = 0
        targets = np.unique(mask)
        targets = [t for t in targets if t != 0]
        for t in targets:
            if t != cls + 1:
                mask[mask == t] = 0
        mask = np.clip(mask, 0, 1)
        if self.mask_shape is not None:
            mask = pil_to_tensor(resize(to_pil_image(mask * 255), self.mask_shape))
            mask[mask > 0] = 1
            return self.img_transform(img), mask, cls
        else:
            return self.img_transform(img), np.expand_dims(mask, axis=0), cls

    def get_denormalizer(self):
        return self.denormalizer

    def get_class_name_by_index_dict(self):
        return {i: self.all_classes[i] for i in range(20)}

